from modelscope import snapshot_download

snapshot_download(
    'Qwen/Qwen2.5-7B',
    local_dir='./model/Qwen2.5-7B'
)