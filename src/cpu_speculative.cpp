// MIT. Every accepted token is the target's first maximum after its own history
// penalty. Discarded cache slots are excluded by a checked logical rewind.
#include "cpu_speculative.hpp"
#include "backend.hpp"
#include "common.hpp"
#include "tokenizer.hpp"
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <regex>
#include <stdexcept>
#include <unordered_set>

namespace mt {
class CpuSpeculativeDecoder {
public:
    static bool append(Qwen3Decoder& d,const std::vector<float>& x,int count,std::vector<float>* h) {
        return h && count>0 && x.size()==size_t(count)*d.hp_.hidden && count<=d.max_seq_-d.past_len_ && d.run(x,count,h);
    }
    static bool rewind(Qwen3Decoder& d,int position) {
        if(position<0 || position>d.past_len_)return false;d.past_len_=position;return true;
    }
    static int remaining(const Qwen3Decoder& d) {return d.max_seq_-d.past_len_;}
};
std::vector<int32_t> cpu_ngram_draft(const std::vector<int32_t>& history,int maximum) {
    if(maximum<=0 || history.size()<3)return {};
    for(int n=std::min<int>(16,int(history.size())-1);n>=2;--n) {
        for(int start=int(history.size())-n-1;start>=0;--start) {
            if(std::equal(history.end()-n,history.end(),history.begin()+start)) {
                const auto begin=history.begin()+start+n;
                return {begin,begin+std::min<int>(maximum,int(history.end()-begin))};
            }
        }
    }
    return {};
}
int cpu_greedy_choice(const std::vector<float>& raw,const std::vector<int32_t>& history,float penalty) {
    if(raw.empty())throw std::runtime_error("empty target logits");
    const auto first_max=[](const std::vector<float>& x) {
        int best=0;for(int i=1;i<int(x.size());++i)if(x[i]>x[best])best=i;return best;
    };
    if(penalty<=1.f)return first_max(raw);
    auto adjusted=raw;std::unordered_set<int32_t> recent;
    for(int i=std::max(0,int(history.size())-100);i<int(history.size());++i)recent.insert(history[i]);
    for(int32_t id:recent)if(id>=0 && id<int(adjusted.size()))
        adjusted[id]=adjusted[id]>0 ? adjusted[id]/penalty : adjusted[id]*penalty;
    return first_max(adjusted);
}
namespace {
bool empty_loop(const std::string& text) {
    static const std::regex marker(R"(\[(\d+(?:\.\d+)?)\]\s*\[S\d+\]\s*\[(\d+(?:\.\d+)?)\])");
    std::deque<std::pair<double,double>> run;size_t previous_end=0;
    for(std::sregex_iterator it(text.begin(),text.end(),marker),end;it!=end;++it) {
        const auto& match=*it;auto gap=text.substr(previous_end,size_t(match.position())-previous_end);
        if(gap.find_first_not_of(" \t\r\n")!=std::string::npos)run.clear();
        try{run.emplace_back(std::stod(match[1].str()),std::stod(match[2].str()));}catch(...){run.clear();}
        if(run.size()>12)run.pop_front();previous_end=size_t(match.position()+match.length());
        if(run.size()==12) {double lo=run.front().first,hi=lo;
            for(const auto& pair:run){lo=std::min({lo,pair.first,pair.second});hi=std::max({hi,pair.first,pair.second});}
            if(hi-lo<=2.0)return true;
        }
    }
    return false;
}
}
std::vector<int32_t> cpu_speculative_generate(Qwen3Decoder& dec,ModelLoader& m,
    const std::vector<float>& fused,int seq,int max_new,int eos,const CpuSpeculativeHooks* hooks) {
    std::vector<int32_t> ids;const int hidden=dec.hidden(),vocab=m.config().text_vocab;
    const char* opt=std::getenv("MTD_CPU_OPT");
    if(hidden<=0 || seq<=0 || max_new<=0 || fused.size()!=size_t(hidden)*seq || !opt ||
        (std::atoi(opt)&262160)!=262160 || !ggml_backend_is_cpu(backend()))return ids;
    using Clock=std::chrono::steady_clock;
    double embedding_seconds=0,decoder_seconds=0,logits_seconds=0,draft_seconds=0,rewind_seconds=0;
    size_t target_calls=0,drafted_calls=0,proposed=0,accepted=0,input_work=0,rewinds=0;
    const auto start=Clock::now();std::vector<float> all_hidden;
    if(!dec.prefill(fused,seq,&all_hidden) || all_hidden.size()!=size_t(hidden)*seq)return {};
    const double prefill_seconds=std::chrono::duration<double>(Clock::now()-start).count();
    if(hooks && hooks->prefilled)hooks->prefilled(all_hidden);
    auto phase=Clock::now();std::vector<float> last(all_hidden.end()-hidden,all_hidden.end());
    auto raw=dec.logits_from_hidden(last);
    logits_seconds+=std::chrono::duration<double>(Clock::now()-phase).count();
    if(raw.size()!=size_t(vocab))return {};
    const char* penalty_env=std::getenv("MTD_REPETITION_PENALTY");
    float penalty=penalty_env?float(std::atof(penalty_env)):1.f;penalty=std::max(1.f,std::min(1.5f,penalty));
    const bool guarded=std::getenv("MTD_LOOP_GUARD")!=nullptr;Tokenizer guard_tokenizer;
    if(guarded && !guard_tokenizer.load(m))return {};
    ids.reserve(size_t(max_new));const char* stop=nullptr;
    auto emit=[&](int32_t id) {
        ids.push_back(id);
        if(id==eos)stop="eos";
        else if(int(ids.size())>=max_new)stop="token_limit";
        else if(guarded && ids.size()>=128 && ids.size()%32==0) {
            auto begin=ids.begin()+std::max(0,int(ids.size())-512);
            if(empty_loop(guard_tokenizer.decode({begin,ids.end()})))stop="empty_marker_loop";
        }
    };
    auto commit=[&]() {if(hooks && hooks->committed)hooks->committed(dec,ids,raw,stop!=nullptr);};
    while(!stop) {
        const int seed=cpu_greedy_choice(raw,ids,penalty);emit(seed);
        if(stop){commit();break;}
        const int before=dec.past_len();
        const int budget=std::max(0,std::min({4,max_new-int(ids.size()),CpuSpeculativeDecoder::remaining(dec)-1}));
        phase=Clock::now();auto proposal=hooks && hooks->draft?hooks->draft(ids,budget):cpu_ngram_draft(ids,budget);
        draft_seconds+=std::chrono::duration<double>(Clock::now()-phase).count();
        if(proposal.size()>size_t(budget))proposal.resize(size_t(budget));
        for(int32_t id:proposal)if(id<0 || id>=vocab)throw std::runtime_error("invalid draft token");
        std::vector<int32_t> inputs{seed};inputs.insert(inputs.end(),proposal.begin(),proposal.end());
        phase=Clock::now();std::vector<float> embeds;
        if(!embed_rows_f32(m.tensor("token_embd.weight"),inputs.data(),int(inputs.size()),hidden,&embeds))break;
        embedding_seconds+=std::chrono::duration<double>(Clock::now()-phase).count();
        phase=Clock::now();std::vector<float> h;
        if(!CpuSpeculativeDecoder::append(dec,embeds,int(inputs.size()),&h) || h.size()!=inputs.size()*hidden)break;
        decoder_seconds+=std::chrono::duration<double>(Clock::now()-phase).count();
        ++target_calls;drafted_calls+=!proposal.empty();proposed+=proposal.size();input_work+=inputs.size();
        if(hooks && hooks->appended)hooks->appended(inputs,h);
        auto logits_row=[&](int index) {
            const auto row_start=h.begin()+size_t(index)*hidden;phase=Clock::now();
            raw=dec.logits_from_hidden({row_start,row_start+hidden});
            logits_seconds+=std::chrono::duration<double>(Clock::now()-phase).count();
            if(raw.size()!=size_t(vocab))throw std::runtime_error("target full logits shape");
        };
        logits_row(0);int keep=1;
        for(size_t i=0;i<proposal.size();++i) {
            if(proposal[i]!=cpu_greedy_choice(raw,ids,penalty))break;
            ++accepted;emit(proposal[i]);
            if(stop)break; // Serial does not evaluate its terminating token.
            ++keep;logits_row(int(i)+1);
        }
        const int position=before+keep;
        if(dec.past_len()!=position) {
            phase=Clock::now();if(!CpuSpeculativeDecoder::rewind(dec,position))throw std::runtime_error("invalid speculative rewind");
            rewind_seconds+=std::chrono::duration<double>(Clock::now()-phase).count();++rewinds;
        }
        commit();
    }
    MT_LOGI("BENCH_GENERATION tokens=%zu stop=%s",ids.size(),stop?stop:"error");
    MT_LOGI("CPU_PROFILE prefill=%.6f embedding=%.6f decoder=%.6f logits=%.6f",prefill_seconds,embedding_seconds,decoder_seconds,logits_seconds);
    MT_LOGI("CPU_SPECULATIVE targetCalls=%zu draftedCalls=%zu proposed=%zu accepted=%zu targetInputTokens=%zu rewinds=%zu draftSeconds=%.9f rewindSeconds=%.9f maximumDraft=4",target_calls,drafted_calls,proposed,accepted,input_work,rewinds,draft_seconds,rewind_seconds);
    const char* trace=std::getenv("MTD_TRACE_TOKENS");
    if(trace && !std::strcmp(trace,"1"))for(size_t offset=0;offset<ids.size();offset+=128) {
        std::string values="[";for(size_t i=offset;i<std::min(ids.size(),offset+128);++i){if(i!=offset)values+=',';values+=std::to_string(ids[i]);}values+=']';
        MT_LOGI("BENCH_TOKEN_TRACE eos=%d total=%zu offset=%zu ids=%s",eos,ids.size(),offset,values.c_str());
    }
    return ids;
}
}
