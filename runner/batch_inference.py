# Copyright 2024 ByteDance and/or its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import logging
import os
import tempfile
import time
import uuid
from pathlib import Path
from typing import List, Optional, Union

import click
import tqdm
from Bio import SeqIO
from ml_collections.config_dict import ConfigDict
from rdkit import Chem

from configs.configs_base import configs as configs_base
from configs.configs_data import data_configs
from configs.configs_inference import inference_configs
from configs.configs_model_type import model_configs
from protenix.config import parse_configs
from protenix.data.json_maker import cif_to_input_json
from protenix.data.json_parser import lig_file_to_atom_info
from protenix.data.utils import pdb_to_cif
from protenix.utils.logger import get_logger
from runner.inference import (
    InferenceRunner,
    download_infercence_cache,
    infer_predict,
    update_gpu_compatible_configs,
)
from runner.msa_search import msa_search, update_infer_json


def _import_protenixscore_runner():
    try:
        from protenixscore.score import run_score
        return run_score
    except ImportError:
        # Try repo-local import when running from source checkout
        import sys
        from pathlib import Path

        repo_root = Path(__file__).resolve().parents[2]
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))
        from protenixscore.score import run_score  # type: ignore

        return run_score

logger = get_logger(__name__)


def init_logging():
    LOG_FORMAT = "%(asctime)s,%(msecs)-3d %(levelname)-8s [%(filename)s:%(lineno)s %(funcName)s] %(message)s"
    logging.basicConfig(
        format=LOG_FORMAT,
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
        filemode="w",
    )


def generate_infer_jsons(protein_msa_res: dict, ligand_file: str) -> List[str]:
    protein_chains = []
    if len(protein_msa_res) <= 0:
        raise RuntimeError(f"invalid `protein_msa_res` data in {protein_msa_res}")
    for key, value in protein_msa_res.items():
        protein_chain = {}
        protein_chain["proteinChain"] = {}
        protein_chain["proteinChain"]["sequence"] = key
        protein_chain["proteinChain"]["count"] = value.get("count", 1)
        protein_chain["proteinChain"]["msa"] = value
        protein_chains.append(protein_chain)
    if os.path.isdir(ligand_file):
        ligand_files = [
            str(file) for file in Path(ligand_file).rglob("*") if file.is_file()
        ]
        if len(ligand_files) == 0:
            raise RuntimeError(
                f"can not read a valid `sdf` or `smi` ligand_file in {ligand_file}"
            )
    elif os.path.isfile(ligand_file):
        ligand_files = [ligand_file]
    else:
        raise RuntimeError(f"can not read a special ligand_file: {ligand_file}")

    invalid_ligand_files = []
    sdf_ligand_files = []
    smi_ligand_files = []
    tmp_json_name = uuid.uuid4().hex
    current_local_dir = (
        f"/tmp/{time.strftime('%Y-%m-%d', time.localtime())}/{tmp_json_name}"
    )
    current_local_json_dir = (
        f"/tmp/{time.strftime('%Y-%m-%d', time.localtime())}/{tmp_json_name}_jsons"
    )
    os.makedirs(current_local_dir, exist_ok=True)
    os.makedirs(current_local_json_dir, exist_ok=True)
    for li_file in ligand_files:
        try:
            if li_file.endswith(".smi"):
                smi_ligand_files.append(li_file)
            elif li_file.endswith(".sdf"):
                suppl = Chem.SDMolSupplier(li_file)
                if len(suppl) <= 1:
                    lig_file_to_atom_info(li_file)
                    sdf_ligand_files.append([li_file])
                else:
                    sdf_basename = os.path.join(
                        current_local_dir, os.path.basename(li_file).split(".")[0]
                    )
                    li_files = []
                    for idx, mol in enumerate(suppl):
                        p_sdf_path = f"{sdf_basename}_part_{idx}.sdf"
                        writer = Chem.SDWriter(p_sdf_path)
                        writer.write(mol)
                        writer.close()
                        li_files.append(p_sdf_path)
                        lig_file_to_atom_info(p_sdf_path)
                    sdf_ligand_files.append(li_files)
            else:
                lig_file_to_atom_info(li_file)
                sdf_ligand_files.append(li_file)
        except Exception as exc:
            logging.info(f" lig_file_to_atom_info failed with error info: {exc}")
            invalid_ligand_files.append(li_file)
    logger.info(f"the json to infer will be save to {current_local_json_dir}")
    infer_json_files = []
    for li_files in sdf_ligand_files:
        one_infer_seq = protein_chains[:]
        for li_file in li_files:
            ligand_name = os.path.basename(li_file).split(".")[0]
            ligand_chain = {}
            ligand_chain["ligand"] = {}
            ligand_chain["ligand"]["ligand"] = f"FILE_{li_file}"
            ligand_chain["ligand"]["count"] = 1
            one_infer_seq.append(ligand_chain)
        one_infer_json = [{"sequences": one_infer_seq, "name": ligand_name}]
        json_file_name = os.path.join(
            current_local_json_dir, f"{ligand_name}_sdf_{uuid.uuid4().hex}.json"
        )
        with open(json_file_name, "w") as f:
            json.dump(one_infer_json, f, indent=4)
        infer_json_files.append(json_file_name)

    for smi_ligand_file in smi_ligand_files:
        with open(smi_ligand_file, "r") as f:
            smile_list = f.readlines()
        one_infer_seq = protein_chains[:]
        ligand_name = os.path.basename(smi_ligand_file).split(".")[0]
        for smile in smile_list:
            normalize_smile = smile.replace("\n", "")
            ligand_chain = {}
            ligand_chain["ligand"] = {}
            ligand_chain["ligand"]["ligand"] = normalize_smile
            ligand_chain["ligand"]["count"] = 1
            one_infer_seq.append(ligand_chain)
        one_infer_json = [{"sequences": one_infer_seq, "name": ligand_name}]
        json_file_name = os.path.join(
            current_local_json_dir, f"{ligand_name}_smi_{uuid.uuid4().hex}.json"
        )
        with open(json_file_name, "w") as f:
            json.dump(one_infer_json, f, indent=4)
        infer_json_files.append(json_file_name)
    if len(invalid_ligand_files) > 0:
        logger.warning(
            f"{len(invalid_ligand_files)} sdf file is invaild, one of them is {invalid_ligand_files[0]}"
        )
    return infer_json_files


def get_default_runner(
    seeds: Optional[list] = None,
    n_cycle: int = 10,
    n_step: int = 200,
    n_sample: int = 5,
    dtype: str = "bf16",
    model_name: str = "protenix_base_default_v0.5.0",
    use_msa: bool = True,
    trimul_kernel="cuequivariance",
    triatt_kernel="triattention",
    enable_cache=True,
    enable_fusion=True,
    enable_tf32=True,
) -> InferenceRunner:
    inference_configs["model_name"] = model_name
    configs = {**configs_base, **{"data": data_configs}, **inference_configs}
    configs = parse_configs(
        configs=configs,
        fill_required_with_null=True,
    )
    if seeds is not None:
        configs.seeds = seeds
    model_name = configs.model_name
    _, model_size, model_feature, model_version = model_name.split("_")
    model_specfics_configs = ConfigDict(model_configs[model_name])
    # update model specific configs
    configs.update(model_specfics_configs)
    # the user input configs has the highest priority
    configs.model.N_cycle = n_cycle
    configs.sample_diffusion.N_sample = n_sample
    configs.sample_diffusion.N_step = n_step
    configs.dtype = dtype
    configs.use_msa = use_msa
    configs.triangle_multiplicative = trimul_kernel
    configs.triangle_attention = triatt_kernel
    configs.enable_diffusion_shared_vars_cache = enable_cache
    configs.enable_efficient_fusion = enable_fusion
    configs.enable_tf32 = enable_tf32

    configs = update_gpu_compatible_configs(configs)
    logger.info(
        f"Inference by Protenix: model_size: {model_size}, with_feature: {model_feature.replace('-', ',')}, model_version: {model_version}, dtype: {configs.dtype}"
    )
    logger.info(
        f"Triangle_multiplicative kernel: {trimul_kernel}, Triangle_attention kernel: {triatt_kernel}"
    )
    logger.info(
        f"enable_diffusion_shared_vars_cache: {configs.enable_diffusion_shared_vars_cache}, "
        + f"enable_efficient_fusion: {configs.enable_efficient_fusion}, enable_tf32: {configs.enable_tf32}"
    )
    download_infercence_cache(configs)
    return InferenceRunner(configs)


def inference_jsons(
    json_file: str,
    out_dir: str = "./output",
    use_msa: bool = True,
    seeds: list = [101],
    n_cycle: int = 10,
    n_step: int = 200,
    n_sample: int = 5,
    dtype: str = "bf16",
    model_name: str = "protenix_base_default_v0.5.0",
    trimul_kernel="cuequivariance",
    triatt_kernel="triattention",
    enable_cache=True,
    enable_fusion=True,
    enable_tf32=True,
    msa_server_mode: str = "protenix",
) -> None:
    """
    infer_json: json file or directory, will run infer with these jsons

    """
    infer_jsons = []
    if os.path.isdir(json_file):
        infer_jsons = [
            str(file) for file in Path(json_file).rglob("*") if file.is_file()
        ]
        if len(infer_jsons) == 0:
            raise RuntimeError(
                f"can not read a valid `sdf` or `smi` ligand_file in {json_file}"
            )
    elif os.path.isfile(json_file):
        infer_jsons = [json_file]
    else:
        raise RuntimeError(f"can not read a special ligand_file: {json_file}")
    infer_jsons = [file for file in infer_jsons if file.endswith(".json")]
    logger.info(f"will infer with {len(infer_jsons)} jsons")
    if len(infer_jsons) == 0:
        return

    infer_errors = {}
    inference_configs["dump_dir"] = out_dir
    inference_configs["input_json_path"] = infer_jsons[0]
    runner = get_default_runner(
        seeds,
        n_cycle,
        n_step,
        n_sample,
        dtype,
        model_name,
        use_msa,
        trimul_kernel,
        triatt_kernel,
        enable_cache,
        enable_fusion,
        enable_tf32,
    )
    configs = runner.configs
    for idx, infer_json in enumerate(tqdm.tqdm(infer_jsons)):
        try:
            configs["input_json_path"] = update_infer_json(
                infer_json, out_dir=out_dir, use_msa=use_msa, mode=msa_server_mode
            )
            infer_predict(runner, configs)
        except Exception as exc:
            infer_errors[infer_json] = str(exc)
    if len(infer_errors) > 0:
        logger.warning(f"run inference failed: {infer_errors}")


@click.group()
def protenix_cli():
    return


@click.command()
@click.option("-i", "--input", type=str, help="json files or dir for inference")
@click.option("-o", "--out_dir", default="./output", type=str, help="infer result dir")
@click.option(
    "-s", "--seeds", type=str, default="101", help="the inference seed, split by comma"
)
@click.option("-c", "--cycle", type=int, default=10, help="pairformer cycle number")
@click.option("-p", "--step", type=int, default=200, help="diffusion step")
@click.option("-e", "--sample", type=int, default=5, help="sample number")
@click.option("-d", "--dtype", type=str, default="bf16", help="inference dtype")
@click.option(
    "-n",
    "--model_name",
    type=str,
    default="protenix_base_default_v0.5.0",
    help="select model checkpoint for inference",
)
@click.option(
    "--use_msa",
    type=bool,
    default=True,
    help="use msa for inference or not, use the precomputed msa or msa searching by server",
)
@click.option(
    "--use_default_params", type=bool, default=True, help="use the default params"
)
@click.option(
    "--trimul_kernel",
    type=str,
    default="cuequivariance",
    help="Kernel to use for triangle multiplicative update. Options: 'cuequivariance', 'torch'.",
)
@click.option(
    "--triatt_kernel",
    type=str,
    default="triattention",
    help="Kernel to use for triangle attention. Options: 'triattention', 'cuequivariance', 'deepspeed', 'torch'.",
)
@click.option(
    "--enable_cache",
    type=bool,
    default=True,
    help="The diffusion module precomputes and caches pair_z, p_lm, and c_l (which are shareable across the N_sample and N_step dimensions)",
)
@click.option(
    "--enable_fusion",
    type=bool,
    default=True,
    help="The diffusion transformer consists of 24 transformer blocks, and the biases in these blocks can be pre-transformed in terms of dimensionality and normalization",
)
@click.option(
    "--enable_tf32",
    type=bool,
    default=True,
    help="When the diffusion module uses FP32 computation, enabling enable_tf32 reduces the matrix multiplication precision from FP32 to TF32.",
)
@click.option(
    "--msa_server_mode",
    type=str,
    default="protenix",
    help="msa search mode, protenix or colabfold",
)
def predict(
    input,
    out_dir,
    seeds,
    cycle,
    step,
    sample,
    dtype,
    model_name,
    use_msa,
    use_default_params,
    trimul_kernel,
    triatt_kernel,
    enable_cache,
    enable_fusion,
    enable_tf32,
    msa_server_mode,
):
    """
    predict: Run predictions with protenix.
    :param input, out_dir, seeds, cycle, step, sample, model_name, use_msa, use_default_params, trimul_kernel, triatt_kernel
    :return:
    """
    init_logging()
    logger.info(f"run infer with input={input}, out_dir={out_dir}, sample={sample}")
    if use_default_params:
        if model_name in [
            "protenix_base_default_v0.5.0",
            "protenix_base_constraint_v0.5.0",
        ]:
            cycle = 10
            step = 200
        elif model_name in [
            "protenix_mini_esm_v0.5.0",
            "protenix_mini_ism_v0.5.0",
            "protenix_mini_default_v0.5.0",
            "protenix_tiny_default_v0.5.0",
        ]:
            cycle = 4
            step = 5
            if model_name in [
                "protenix_mini_esm_v0.5.0",
                "protenix_mini_ism_v0.5.0",
            ]:
                use_msa = False
        else:
            raise RuntimeError(
                f"{model_name} is not supported for inference in our model list"
            )
    logger.info(
        f"Using the default params for inference for model {model_name}: cycle={cycle}, step={step}, use_msa={use_msa}"
    )
    assert trimul_kernel in [
        "cuequivariance",
        "torch",
    ], "Kernel to use for triangle multiplicative update. Options: 'cuequivariance', 'torch'."
    assert triatt_kernel in [
        "triattention",
        "cuequivariance",
        "deepspeed",
        "torch",
    ], "Kernel to use for triangle attention. Options: 'triattention', 'cuequivariance', 'deepspeed', 'torch'."
    seeds = list(map(int, seeds.split(",")))
    inference_jsons(
        input,
        out_dir,
        use_msa,
        seeds=seeds,
        n_cycle=cycle,
        n_step=step,
        n_sample=sample,
        dtype=dtype,
        model_name=model_name,
        trimul_kernel=trimul_kernel,
        triatt_kernel=triatt_kernel,
        enable_cache=enable_cache,
        enable_fusion=enable_fusion,
        enable_tf32=enable_tf32,
        msa_server_mode=msa_server_mode,
    )


@click.command()
@click.option(
    "--input", type=str, help="pdb or cif files to generate jsons for inference"
)
@click.option("--out_dir", type=str, default="./output", help="dir to save json files")
@click.option(
    "--altloc",
    default="first",
    type=str,
    help=" Select the first altloc conformation of each residue in the input file, \
        or specify the altloc letter for selection. For example, 'first', 'A', 'B', etc.",
)
@click.option(
    "--assembly_id",
    default=None,
    type=str,
    help="Extends the structure based on the Assembly ID in \
                        the input file. The default is no extension",
)
def tojson(input, out_dir="./output", altloc="first", assembly_id=None):
    """
    tojson: convert pdb/cif files or dir to json files for predict.
    :param input, out_dir, altloc, assembly_id
    :return:
    """
    init_logging()
    logger.info(
        f"run tojson with input={input}, out_dir={out_dir}, altloc={altloc}, assembly_id={assembly_id}"
    )
    input_files = []
    if not os.path.exists(input):
        raise RuntimeError(f"input file {input} not exists.")
    if os.path.isdir(input):
        input_files.extend(
            [str(file) for file in Path(input).rglob("*") if file.is_file()]
        )
    elif os.path.isfile(input):
        input_files.append(input)
    else:
        raise RuntimeError(f"can not read a special file: {input}")

    input_files = [
        file for file in input_files if file.endswith(".pdb") or file.endswith(".cif")
    ]
    if len(input_files) == 0:
        raise RuntimeError(f"can not read a valid `pdb` or `cif` file from {input}")
    logger.info(
        f"will tojson jsons for {len(input_files)} input files with `pdb` or `cif` format."
    )
    output_jsons = []
    os.makedirs(out_dir, exist_ok=True)
    for input_file in input_files:
        stem, _ = os.path.splitext(os.path.basename(input_file))
        pdb_name = stem[:20]
        output_json = os.path.join(out_dir, f"{pdb_name}-{uuid.uuid4().hex}.json")
        if input_file.endswith(".pdb"):
            with tempfile.NamedTemporaryFile(suffix=".cif") as tmp:
                tmp_cif_file = tmp.name
                pdb_to_cif(input_file, tmp_cif_file)
                cif_to_input_json(
                    tmp_cif_file,
                    assembly_id=assembly_id,
                    altloc=altloc,
                    sample_name=pdb_name,
                    output_json=output_json,
                )
        elif input_file.endswith(".cif"):
            cif_to_input_json(
                input_file,
                assembly_id=assembly_id,
                altloc=altloc,
                output_json=output_json,
            )
        else:
            raise RuntimeError(f"can not read a special ligand_file: {input_file}")
        output_jsons.append(output_json)
    logger.info(f"{len(output_jsons)} generated jsons have been saved to {out_dir}.")
    return output_jsons


@click.command()
@click.option(
    "--input", type=str, help="file to do msa search, support `json` or `fasta` format"
)
@click.option("--out_dir", type=str, default="./output", help="dir to save msa results")
@click.option(
    "--msa_server_mode",
    type=str,
    default="protenix",
    help="msa search mode, protenix or colabfold",
)
def msa(input, out_dir, msa_server_mode) -> Union[str, dict]:
    """
    msa: do msa search by mmseqs. If input is in `fasta`, it should all be proteinChain.
    :param input, out_dir, msa_server_mode
    :return:
    """
    init_logging()
    logger.info(f"run msa with input={input}, out_dir={out_dir}")
    if input.endswith(".json"):
        msa_input_json = update_infer_json(
            input, out_dir, use_msa=True, mode=msa_server_mode
        )
        logger.info(f"msa results have been update to {msa_input_json}")
        return msa_input_json
    elif input.endswith(".fasta"):
        records = list(SeqIO.parse(input, "fasta"))
        protein_seqs = []
        for seq in records:
            protein_seqs.append(str(seq.seq))
        protein_seqs = sorted(protein_seqs)
        msa_res_subdirs = msa_search(protein_seqs, out_dir, msa_server_mode)
        assert len(msa_res_subdirs) == len(msa_res_subdirs), "msa search failed"
        fasta_msa_res = dict(zip(protein_seqs, msa_res_subdirs))
        logger.info(
            f"msa result is: {fasta_msa_res}, and it has been save to {out_dir}"
        )
        return fasta_msa_res
    else:
        raise RuntimeError(f"only support `json` or `fasta` format, but got : {input}")


@click.command()
@click.option("--input", type=str, required=True, help="pdb/cif files or dir to score")
@click.option("--output", type=str, required=True, help="output directory")
@click.option("--recursive", is_flag=True, default=False, help="recurse into subdirectories")
@click.option("--glob", type=str, default="*.pdb,*.cif", help="comma-separated glob patterns")
@click.option("--use_msa/--no-use_msa", default=False, help="enable MSA features")
@click.option("--use_esm/--no-use_esm", default=False, help="enable ESM features")
@click.option("--convert_pdb_to_cif/--no_convert_pdb_to_cif", default=True, help="convert PDB inputs to CIF")
@click.option("--keep_intermediate/--no_keep_intermediate", default=False, help="keep intermediate files")
@click.option("--intermediate_dir", type=str, default=None, help="intermediate directory")
@click.option("--assembly_id", type=str, default=None, help="assembly id for mmCIF expansion")
@click.option("--altloc", type=str, default="first", help="altloc selection")
@click.option("--checkpoint_dir", type=str, default=None, help="checkpoint directory")
@click.option(
    "--model_name",
    type=str,
    default="protenix_base_default_v0.5.0",
    help="model name",
)
@click.option("--device", type=str, default="auto", help="cpu|cuda:N|auto")
@click.option("--dtype", type=str, default="bf16", help="fp32|bf16|fp16")
@click.option("--num_workers", type=int, default=4, help="dataloader workers")
@click.option("--batch_size", type=int, default=1, help="batch size (currently 1)")
@click.option("--max_tokens", type=int, default=None, help="max tokens")
@click.option("--max_atoms", type=int, default=None, help="max atoms")
@click.option("--summary_format", type=str, default="json", help="json|csv")
@click.option("--aggregate_csv", type=str, default=None, help="aggregate csv output")
@click.option("--overwrite/--no_overwrite", default=False, help="overwrite outputs")
@click.option("--failed_log", type=str, default=None, help="failed records log")
@click.option(
    "--missing_atom_policy",
    type=str,
    default="reference",
    help="reference|zero|error",
)
def score(
    input,
    output,
    recursive,
    glob,
    use_msa,
    use_esm,
    convert_pdb_to_cif,
    keep_intermediate,
    intermediate_dir,
    assembly_id,
    altloc,
    checkpoint_dir,
    model_name,
    device,
    dtype,
    num_workers,
    batch_size,
    max_tokens,
    max_atoms,
    summary_format,
    aggregate_csv,
    overwrite,
    failed_log,
    missing_atom_policy,
):
    """
    score: Score existing structures using the Protenix confidence head.
    """
    run_score = _import_protenixscore_runner()
    from argparse import Namespace

    args = Namespace(
        command="score",
        input=input,
        output=output,
        recursive=recursive,
        glob=glob,
        score_only=True,
        use_msa=use_msa,
        use_esm=use_esm,
        convert_pdb_to_cif=convert_pdb_to_cif,
        keep_intermediate=keep_intermediate,
        intermediate_dir=intermediate_dir,
        assembly_id=assembly_id,
        altloc=altloc,
        checkpoint_dir=checkpoint_dir,
        model_name=model_name,
        device=device,
        dtype=dtype,
        num_workers=num_workers,
        batch_size=batch_size,
        max_tokens=max_tokens,
        max_atoms=max_atoms,
        write_full_confidence=True,
        write_summary_confidence=True,
        summary_format=summary_format,
        aggregate_csv=aggregate_csv,
        overwrite=overwrite,
        failed_log=failed_log,
        missing_atom_policy=missing_atom_policy,
    )
    run_score(args)


protenix_cli.add_command(predict)
protenix_cli.add_command(tojson)
protenix_cli.add_command(msa)
protenix_cli.add_command(score)


if __name__ == "__main__":
    predict()
