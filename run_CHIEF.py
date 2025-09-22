#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CHIEF Protein Design CLI

Usage examples
--------------
# Use defaults (./test/MDH.pdb -> ./test/designed_sequence.fasta, 10 sequences)
python run_CHIEF.py

# Custom inputs
python run_CHIEF.py \
  --pdb_path ./inputs/target.pdb \
  --num_seqs 100 \
  --out_fasta ./outputs/target_designs.fasta \
  --temperature 0.15 \
  --device cuda
"""

import argparse
import logging
import os
from pathlib import Path
from datetime import datetime
from typing import List

import torch
from tqdm.auto import tqdm


from model.ProteinMPNN.proteinmpnnwrapper import ProteinMPNNWrapper
from model.esm.esmifwrapper import ESMIFWrapper
from model.PiFold.pifoldwrapper import PiFoldWrapper
from model.Frame2seq.frame2seqwrapper import Frame2SeqWrapper
from model.CHIEF.chiefwrapper import ChiefWrapper
from utils.data_utils import PDBparser, featurize


ALPHABET = "ACDEFGHIKLMNPQRSTVWYX"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Protein sequence design with CHIEF ensemble."
    )
    parser.add_argument(
        "--pdb_path",
        type=str,
        default="./test/MDH.pdb",
        help="Path to the input PDB file (default: ./test/MDH.pdb)",
    )
    parser.add_argument(
        "--num_seqs",
        type=int,
        default=10,
        help="Number of sequences to sample (default: 10)",
    )
    parser.add_argument(
        "--out_fasta",
        type=str,
        default="./test/designed_sequence.fasta",
        help="Output FASTA file path (default: ./test/designed_sequence.fasta)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature passed to CHIEF.sample (default: 1.0)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cuda", "cpu"],
        help="Computation device override. Defaults to 'cuda' if available, else 'cpu'.",
    )
    return parser.parse_args()


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def resolve_device(cli_choice: str | None) -> str:
    if cli_choice is not None:
        if cli_choice == "cuda" and not torch.cuda.is_available():
            logging.warning("CUDA requested but not available. Falling back to CPU.")
            return "cpu"
        return cli_choice
    # Default behavior
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_models(device: str) -> ChiefWrapper:
    """
    Instantiate all base models and the CHIEF ensemble on the selected device.
    """
    logging.info(f"Initializing models on device: {device}")

    # NOTE: Adjust checkpoint paths as needed for your environment
    proteinmpnn_vanilla = ProteinMPNNWrapper(
        "./checkpoints/proteinmpnn/proteinmpnn_vanilla_v_48_020.pt", device=device
    )
    proteinmpnn_soluble = ProteinMPNNWrapper(
        "./checkpoints/proteinmpnn/proteinmpnn_soluble_v_48_020.pt", device=device
    )
    esmif = ESMIFWrapper(device=device)
    frame2seq = Frame2SeqWrapper(device=device)
    pifold = PiFoldWrapper(device=device)

    chief = ChiefWrapper(
        model_dict={
            "ProteinMPNN_vanilla": proteinmpnn_vanilla,
            "ProteinMPNN_soluble": proteinmpnn_soluble,
            "ESM-IF": esmif,
            "Frame2seq": frame2seq,
            "PiFold": pifold,
        },
        base_models_list=[
            "ProteinMPNN_vanilla",
            "ProteinMPNN_soluble",
            "ESM-IF",
            "Frame2seq",
            "PiFold",
        ],
        state_dict="./checkpoints/chief/chief.pth",
        device=device,
        need_weights=True,
    )

    logging.info("Models initialized successfully.")
    return chief


def design_sequences(
    chief: ChiefWrapper,
    pdb_path: Path,
    num_seqs: int,
    device: str,
    temperature: float = 1.0,
) -> List[str]:
    """
    Run CHIEF sampling to generate designed sequences.
    """
    if not pdb_path.is_file():
        raise FileNotFoundError(f"PDB file not found: {pdb_path}")

    logging.info(f"Parsing PDB: {pdb_path}")
    batch = PDBparser(str(pdb_path))

    logging.info("Featurizing structure...")
    X, S, mask, lengths, chain_M, residue_idx, mask_self, chain_encoding_all = featurize(
        batch, device
    )

    sequences: List[str] = []
    logging.info(f"Sampling {num_seqs} sequences (temperature={temperature})...")
    with torch.no_grad():
        for i in tqdm(range(num_seqs), desc="Designing"):
            S_sampled = chief.sample(
                X,
                S,
                mask,
                chain_M,
                residue_idx,
                chain_encoding_all,
                temperature=temperature,
            )
            # chief.sample may return a batch; we take the first item
            S_sampled = S_sampled[0]
            seq = "".join(ALPHABET[int(aa)] for aa in S_sampled)
            sequences.append(seq)
            logging.debug(f"Sequence {i+1:03d}: {seq}")

    logging.info("Sampling complete.")
    return sequences


def write_fasta(
    sequences: List[str],
    out_fasta: Path,
    source_pdb: Path,
    metadata: dict | None = None,
) -> None:
    """
    Write sequences to a single FASTA file.
    """
    out_fasta.parent.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    title_pdb = source_pdb.name

    meta_lines = []
    if metadata:
        for k, v in metadata.items():
            meta_lines.append(f"{k}={v}")
    meta_str = "; ".join(meta_lines) if meta_lines else ""

    with out_fasta.open("w", encoding="utf-8") as fh:
        for i, seq in enumerate(sequences, start=1):
            header_parts = [
                f"id=CHIEF_SEQ_{i:03d}",
                f"pdb={title_pdb}",
                f"generated={ts}",
            ]
            if meta_str:
                header_parts.append(meta_str)
            header = "|".join(header_parts)
            fh.write(f">{header}\n")
            fh.write(f"{seq}\n")

    logging.info(f"FASTA written: {out_fasta} ({len(sequences)} sequences)")


def main() -> None:
    setup_logging()
    args = parse_args()

    pdb_path = Path(args.pdb_path).expanduser().resolve()
    out_fasta = Path(args.out_fasta).expanduser().resolve()
    num_seqs = int(args.num_seqs)
    temperature = float(args.temperature)

    if num_seqs <= 0:
        raise ValueError("--num_seqs must be a positive integer.")

    device = resolve_device(args.device)
    logging.info(f"Using device: {device}")

    # Load ensemble
    chief = load_models(device=device)

    # Run design
    sequences = design_sequences(
        chief=chief,
        pdb_path=pdb_path,
        num_seqs=num_seqs,
        device=device,
        temperature=temperature,
    )

    # Save outputs
    metadata = {
        "temperature": temperature,
        "device": device,
        "models": "ProteinMPNN_vanilla,ProteinMPNN_soluble,ESM-IF,Frame2seq,PiFold",
        "ensemble": "CHIEF",
    }
    write_fasta(sequences, out_fasta, source_pdb=pdb_path, metadata=metadata)

    # Also print to console for quick inspection
    logging.info("Designed sequences (first few):")
    preview = min(5, len(sequences))
    for i in range(preview):
        logging.info(f"[{i+1:03d}] {sequences[i]}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.exception(f"Fatal error: {e}")
        raise
