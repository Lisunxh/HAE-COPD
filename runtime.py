
import os, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent
def configure():
    sys.dont_write_bytecode = True
    os.environ['PYTHONDONTWRITEBYTECODE']='1'
    for key, sub in {'TMP':'tmp','TEMP':'tmp','TMPDIR':'tmp','MPLCONFIGDIR':'mpl','TORCH_HOME':'torch','XDG_CACHE_HOME':'general','CUDA_CACHE_PATH':'cuda','TORCHINDUCTOR_CACHE_DIR':'inductor','TRITON_CACHE_DIR':'triton','HF_HOME':'hf'}.items():
        p=ROOT/'.runtime'/sub;p.mkdir(parents=True,exist_ok=True);os.environ[key]=str(p)
    def allowed(p): return isinstance(p,int) or Path(os.fsdecode(p)).resolve().is_relative_to(ROOT)
    def guard(event,args):
        if event=='open' and args[2] & (os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND) and not allowed(args[0]): raise PermissionError('Output outside package prohibited')
        if event in ('os.remove','os.rmdir','os.mkdir','os.chmod','os.utime') and not allowed(args[0]): raise PermissionError('Mutation outside package prohibited')
        if event in ('os.rename','os.link','os.symlink') and not all(allowed(p) for p in args[:2]): raise PermissionError('Mutation outside package prohibited')
    sys.addaudithook(guard)
    import torch
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=True
    torch.backends.cudnn.benchmark=False
    return ROOT

