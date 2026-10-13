import pytest
import moss_remote

def valid_identity():
    return {'modelSha256':moss_remote.MODEL_SHA256,'sourceCommit':moss_remote.SOURCE_COMMIT,
            'policy':moss_remote.POLICY,'device':'cpu','quantization':'q8_0','threads':16,
            'cpuOpt':48,'binarySha256':moss_remote.VALIDATED_BINARY_SHA256}

@pytest.mark.parametrize('field,value',[('cpuOpt',0),('cpuOpt',786480),('binarySha256','0'*64),
                                      ('threads',32),('sourceCommit','0'*40)])
def test_rejects_unvalidated_runtime(field,value):
    identity=valid_identity();identity[field]=value
    with pytest.raises(RuntimeError,match='pinned identity'):moss_remote._validate_identity(identity)

def test_accepts_validated_runtime_and_configured_build(monkeypatch):
    monkeypatch.delenv('MOSS_API_EXPECTED_BINARY_SHA256',raising=False)
    moss_remote._validate_identity(valid_identity())
    monkeypatch.setenv('MOSS_API_EXPECTED_BINARY_SHA256','a'*64)
    moss_remote._validate_identity({**valid_identity(),'binarySha256':'a'*64})

def test_completed_cache_contract_is_unchanged():
    origin='190a569c13b4b247450f2fb3b2a431244e84833e'
    assert moss_remote.signature()==f'remote-cpu:{origin}:{moss_remote.MODEL_SHA256}:{moss_remote.POLICY}'
    assert moss_remote.legacy_signature()==f'remote-cpu:{origin}:{moss_remote.MODEL_SHA256}:moss-cpp-cpu-eos-v1'
