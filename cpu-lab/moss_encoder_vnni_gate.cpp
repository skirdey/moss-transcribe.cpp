// MIT. Real-weight, every-chunk encoder bit gate. Only numeric output is public.
#include "audio_io.hpp"
#include "mel.hpp"
#include "whisper_encoder.hpp"
#include "cpu_encoder_q8.hpp"
#include "cpu_profile.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <vector>
namespace {
void require(bool b,const char* m) { if(!b)throw std::runtime_error(m); }
void opt(const char* v) {
#ifdef _WIN32
    _putenv_s("MTD_CPU_OPT",v);
#else
    setenv("MTD_CPU_OPT",v,1);
#endif
}
}
int main(int argc,char** argv) {
    try {
        require(argc>=3,"usage: moss_encoder_vnni_gate MODEL AUDIO...");
        const char* candidate=std::getenv("MTD_ENCODER_GATE_OPT");
        if(!candidate)candidate="65584";
        require(!std::strcmp(candidate,"65584") || !std::strcmp(candidate,"196656"),"supported gate candidate");
        if(!mt::CpuEncoderQ8::supported())return 77;
        mt::ModelLoader model;require(model.load(argv[1]),"model load");model.promote_small_f16_to_f32();
        mt::WhisperMel mel(model);mt::WhisperEncoder encoder(model);
        const size_t chunk_size=model.config().feat_n_samples;require(chunk_size>0,"chunk size");
        size_t total_bits=0,chunks=0;
        for(int index=2;index<argc;++index) {
            mt::Audio audio;require(mt::load_audio_16k_mono(argv[index],audio) && !audio.samples.empty(),"audio load");
            for(size_t off=0;off<audio.samples.size();off+=chunk_size) {
                std::vector<float> chunk(audio.samples.begin()+off,audio.samples.begin()+std::min(off+chunk_size,audio.samples.size()));
                chunk.resize(chunk_size,0);std::vector<float> feat;int m=0,t=0;mel.compute(chunk,feat,m,t);
                const auto saved_feat=feat;std::vector<float> ref,got;int rt=0,rd=0,ct=0,cd=0;
                opt("48");auto start=std::chrono::steady_clock::now();encoder.encode(feat,m,t,ref,rt,rd);
                const double ref_seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
                mt::cpu_profile_reset();opt(candidate);start=std::chrono::steady_clock::now();encoder.encode(feat,m,t,got,ct,cd);
                const double candidate_seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
                mt::cpu_profile_print();
                require(!ref.empty() && got.size()==ref.size() && rt==ct && rd==cd,"encoder shape");
                size_t bits=0;for(size_t i=0;i<ref.size();++i) {
                    require(std::isfinite(ref[i]) && std::isfinite(got[i]),"finite encoder output");
                    bits+=std::memcmp(&ref[i],&got[i],sizeof(float))!=0;
                }
                require(!std::memcmp(feat.data(),saved_feat.data(),feat.size()*sizeof(float)),"mel immutable");
                total_bits+=bits;++chunks;
                std::printf("{\"audioIndex\":%d,\"chunk\":%zu,\"candidateOpt\":%d,\"T\":%d,\"D\":%d,\"elements\":%zu,\"floatBitDifferences\":%zu,\"referenceEncoderSeconds\":%.9f,\"candidateEncoderSeconds\":%.9f,\"melUnchanged\":true,\"fullInputLatency\":false}\n",index-2,off/chunk_size,std::atoi(candidate),ct,cd,got.size(),bits,ref_seconds,candidate_seconds);
                std::fflush(stdout);require(!bits,"real-weight encoder bit parity");
            }
        }
        require(chunks && !total_bits,"complete every-chunk encoder parity");return 0;
    } catch(const std::exception& e) { std::fprintf(stderr,"encoder real gate: %s\n",e.what());return 1; }
}
