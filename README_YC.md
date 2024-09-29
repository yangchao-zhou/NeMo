https://docs.nvidia.com/nemo-framework/user-guide/latest/playbooks/mistralsft.html#mistralsft-playbook

## Apex安装
git clone https://github.com/NVIDIA/apex.git
cd apex
git checkout $apex_commit
pip install . -v --no-build-isolation --disable-pip-version-check --no-cache-dir --config-settings "--build-option=--cpp_ext --cuda_ext --fast_layer_norm --distributed_adam --deprecated_fused_adam --group_norm"


python3 /opt/NeMo-Framework-Launcher/launcher_scripts/nemo_launcher/collections/dataprep_scripts/dolly_dataprep/preprocess.py --input /mnt/workspace/yangchao.zhou/opt/data/databricks/databricks-dolly-15k/databricks-dolly-15k.jsonl


export HYDRA_FULL_ERROR=1

export TMPDIR=/mnt/workspace/yangchao.zhou/opt/models/tmp
MODEL="/mnt/workspace/yangchao.zhou/opt/models/mistral/Mistral-NeMo-12B-Instruct.nemo"
TRAIN_DS="[/mnt/workspace/yangchao.zhou/opt/data/databricks/databricks-dolly-15k/training.jsonl]"
VALID_DS="[/mnt/workspace/yangchao.zhou/opt/data/databricks/databricks-dolly-15k/validation.jsonl]"
TEST_DS="[/mnt/workspace/yangchao.zhou/opt/data/databricks/databricks-dolly-15k/test.jsonl]"
VALID_NAMES="[databricks-dolly-15k]"
RESULTS="/mnt/workspace/yangchao.zhou/opt/results"
CONCAT_SAMPLING_PROBS="[1]"
TP_SIZE=2
PP_SIZE=4


torchrun --nproc_per_node=8 \
/opt/NeMo/examples/nlp/language_modeling/tuning/megatron_gpt_finetuning.py \
   trainer.precision=bf16 \
   trainer.devices=8 \
   trainer.num_nodes=1 \
   trainer.val_check_interval=0.5 \
   trainer.max_steps=100 \
   model.restore_from_path=${MODEL} \
   model.micro_batch_size=1 \
   model.global_batch_size=32 \
   model.tensor_model_parallel_size=${TP_SIZE} \
   model.pipeline_model_parallel_size=${PP_SIZE} \
   model.megatron_amp_O2=True \
   model.sequence_parallel=True \
   model.activations_checkpoint_granularity=selective \
   model.activations_checkpoint_method=uniform \
   model.optim.name=distributed_fused_adam \
   model.optim.lr=1e-6 \
   model.answer_only_loss=True \
   model.peft.peft_scheme=none \
   model.data.train_ds.file_names=${TRAIN_DS} \
   model.data.validation_ds.file_names=${VALID_DS} \
   model.data.test_ds.file_names=${TEST_DS} \
   model.data.train_ds.concat_sampling_probabilities=${CONCAT_SAMPLING_PROBS} \
   model.data.train_ds.max_seq_length=512 \
   model.data.validation_ds.max_seq_length=512 \
   model.data.train_ds.micro_batch_size=1 \
   model.data.train_ds.global_batch_size=32 \
   model.data.validation_ds.micro_batch_size=1 \
   model.data.validation_ds.global_batch_size=32 \
   model.data.test_ds.micro_batch_size=1 \
   model.data.test_ds.global_batch_size=32 \
   model.data.train_ds.num_workers=0 \
   model.data.validation_ds.num_workers=0 \
   model.data.test_ds.num_workers=0 \
   model.data.validation_ds.metric.name=loss \
   model.data.test_ds.metric.name=loss \
   exp_manager.create_wandb_logger=False \
   exp_manager.explicit_log_dir=${RESULTS} \
   exp_manager.resume_if_exists=True \
   exp_manager.resume_ignore_no_checkpoint=True \
   exp_manager.create_checkpoint_callback=True \
   exp_manager.checkpoint_callback_params.monitor=validation_loss \
   exp_manager.checkpoint_callback_params.save_best_model=False \
   exp_manager.checkpoint_callback_params.save_nemo_on_train_end=True \
   exp_manager.checkpoint_callback_params.mode=min \
   ++cluster_type=BCP
