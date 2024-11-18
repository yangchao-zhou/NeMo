## 镜像
训练镜像：wlcb-aistory-registry-vpc.cn-wulanchabu.cr.aliyuncs.com/base/nemo:24.05.llama3.1
转点镜像：nemo-conver

## 代码更改
注意要替换环境中的代码（/opt），不是本地代码，要更改三个代码
（也可以直接用镜像：nemo-mistral-v2，该环境代码都更改好了，不需要再更改代码）

1. 更改gpt_sft_chat_dataset.py
将/opt/NeMo/nemo/collections/nlp/data/language_modeling/megatron/gpt_sft_chat_dataset.py替换成NeMo/change_file/gpt_sft_chat_dataset.py
2. 更改nemo_model_checkpoint.py
将/opt/NeMo/nemo/utils/callbacks/nemo_model_checkpoint.py替换成NeMo/change_file/nemo_model_checkpoint.py
3. 更改gpt_sft.yaml
将/opt/NeMo-Aligner/examples/nlp/gpt/conf/gpt_sft.yaml替换成NeMo-Aligner/change_file/gpt_sft.yaml

## 训练
加环境变量，在训练脚本中加或者终端输入都可以
```bash
export PYTHONPATH="/opt/NeMo:$PYTHONPATH"
export TMPDIR=/path/to/directory/with/more/space  #改成/mnt/data/.../tmp，这个不改会报错No space left on device: '/tmp/
```
启动训练
```bash
bash change_file/mistral_sft.sh
```
## 转点
用本地的NeMo代码库（git clone的），不要用/opt/NeMo的
加环境变量
```bash
export PYTHONPATH="/opt/NeMo:$PYTHONPATH"
```
启动转点
```bash
cd 你的NeMo && python3 scripts/checkpoint_converters/convert_mistral_7b_nemo_to_hf.py \
     --input_name_or_path results/checkpoints/.nemo文件 \  #替换成.nemo文件
     --output_path results/hf_models/dir \  # 替换成hf_model_dir
     --hf_model_name /mnt/data/nlp_models/mistralai/Mistral-Nemo-Instruct-2407-toconvernemo  # 一定用这个
```

## 评估
hf、vllm推理都可以