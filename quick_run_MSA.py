"""
Before running better to run the following 2 shell commands:
sudo mkdir -m 777 --parents /tmp/ramdisk
sudo mount -t tmpfs -o size=9G ramdisk /tmp/ramdisk

jackhmmer takes a parameter which is the number of iterations (default is 5) consider reducing this if possible

"""

# --- Python imports ---
import collections
import copy
import tqdm
from concurrent import futures
import json
import random

from urllib import request
from matplotlib import gridspec
import matplotlib.pyplot as plt
import numpy as np
import py3Dmol

from alphafold.model import model
from alphafold.model import config
from alphafold.model import data

from alphafold.data import feature_processing
from alphafold.data import msa_pairing
from alphafold.data import parsers
from alphafold.data import pipeline
from alphafold.data import pipeline_multimer
from alphafold.data.tools import jackhmmer

from alphafold.common import protein
import jax
import os
os.environ['TF_FORCE_UNIFIED_MEMORY'] = '1'
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '2.0'

from alphafold.notebooks import notebook_utils

sequence_1 = 'MAAHKGAEHHHKAAEHHEQAAKHHHAAAEHHEKGEHEQAAHHADTAYAHHKHAEEHAAQAAKHDAEHHAPKPH'  #@param {type:"string"}
sequence_2 = ''  #@param {type:"string"}
sequence_3 = ''  #@param {type:"string"}
sequence_4 = ''  #@param {type:"string"}
sequence_5 = ''  #@param {type:"string"}
sequence_6 = ''  #@param {type:"string"}
sequence_7 = ''  #@param {type:"string"}
sequence_8 = ''  #@param {type:"string"}

input_sequences = (sequence_1, sequence_2, sequence_3, sequence_4,
                   sequence_5, sequence_6, sequence_7, sequence_8)

#@markdown If folding a comple
# x target and all the input sequences are
#@markdown prokaryotic then set `is_prokaryotic` to `True`. Set to `False`
#@markdown otherwise or if the origin is unknown.

is_prokaryote = False  #@param {type:"boolean"}

MIN_SINGLE_SEQUENCE_LENGTH = 16
MAX_SINGLE_SEQUENCE_LENGTH = 2500
MAX_MULTIMER_LENGTH = 2500

# Validate the input.
sequences, model_type_to_use = notebook_utils.validate_input(
    input_sequences=input_sequences,
    min_length=MIN_SINGLE_SEQUENCE_LENGTH,
    max_length=MAX_SINGLE_SEQUENCE_LENGTH,
    max_multimer_length=MAX_MULTIMER_LENGTH)

# note that you will probably need to change the jackhmmer binary path  (after installing jackhmmer with command: brew install hmmer)
JACKHMMER_BINARY_PATH = '/usr/local/Cellar/hmmer/3.3.2/bin/jackhmmer'
DB_ROOT_PATH = f'https://storage.googleapis.com/alphafold-colab-europe/latest/'
# The z_value is the number of sequences in a database.
MSA_DATABASES = [
    {'db_name': 'uniref90',
     'db_path': f'{DB_ROOT_PATH}uniref90_2021_03.fasta',
     'num_streamed_chunks': 59,
     'z_value': 135_301_051},
    # {'db_name': 'smallbfd',
    #  'db_path': f'{DB_ROOT_PATH}bfd-first_non_consensus_sequences.fasta',
    #  'num_streamed_chunks': 17,
    #  'z_value': 65_984_053},
    # {'db_name': 'mgnify',
    #  'db_path': f'{DB_ROOT_PATH}mgy_clusters_2019_05.fasta',
    #  'num_streamed_chunks': 71,
    #  'z_value': 304_820_129},
]

# Search UniProt and construct the all_seq features only for heteromers, not homomers.
# if model_type_to_use == notebook_utils.ModelType.MULTIMER and len(set(sequences)) > 1:
#   MSA_DATABASES.extend([
#       # Swiss-Prot and TrEMBL are concatenated together as UniProt.
#       {'db_name': 'uniprot',
#        'db_path': f'{DB_ROOT_PATH}uniprot_2021_03.fasta',
#        'num_streamed_chunks': 98,
#        'z_value': 219_174_961 + 565_254},
#   ])

TOTAL_JACKHMMER_CHUNKS = sum([cfg['num_streamed_chunks'] for cfg in MSA_DATABASES])

MAX_HITS = {
    'uniref90': 10_000,
    'smallbfd': 5_000,
    'mgnify': 501,
    'uniprot': 50_000,
}


def get_msa(fasta_path):
  """Searches for MSA for the given sequence using chunked Jackhmmer search."""

  # Run the search against chunks of genetic databases (since the genetic
  # databases don't fit in Colab disk).
  raw_msa_results = collections.defaultdict(list)
  with tqdm.tqdm(total=TOTAL_JACKHMMER_CHUNKS, ) as pbar:
    def jackhmmer_chunk_callback(i):
      pbar.update(n=1)

    for db_config in MSA_DATABASES:
      db_name = db_config['db_name']
      pbar.set_description(f'Searching {db_name}')
      jackhmmer_runner = jackhmmer.Jackhmmer(
          binary_path=JACKHMMER_BINARY_PATH,
          database_path=db_config['db_path'],
          get_tblout=True,
          num_streamed_chunks=db_config['num_streamed_chunks'],
          streaming_callback=jackhmmer_chunk_callback,
          z_value=db_config['z_value'])
      # Group the results by database name.
      if not os.path.exists('/tmp/ramdisk/'):
          os.mkdir('/tmp/ramdisk/')
      raw_msa_results[db_name].extend(jackhmmer_runner.query(fasta_path))

  return raw_msa_results


features_for_chain = {}
raw_msa_results_for_sequence = {}
for sequence_index, sequence in enumerate(sequences, start=1):
  print(f'\nGetting MSA for sequence {sequence_index}')

  fasta_path = f'target_{sequence_index}.fasta'
  with open(fasta_path, 'wt') as f:
    f.write(f'>query\n{sequence}')

  # Don't do redundant work for multiple copies of the same chain in the multimer.
  if sequence not in raw_msa_results_for_sequence:
    raw_msa_results = get_msa(fasta_path=fasta_path)
    raw_msa_results_for_sequence[sequence] = raw_msa_results
  else:
    raw_msa_results = copy.deepcopy(raw_msa_results_for_sequence[sequence])

  # Extract the MSAs from the Stockholm files.
  # NB: deduplication happens later in pipeline.make_msa_features.
  single_chain_msas = []
  uniprot_msa = None
  for db_name, db_results in raw_msa_results.items():
    merged_msa = notebook_utils.merge_chunked_msa(
        results=db_results, max_hits=MAX_HITS.get(db_name))
    if merged_msa.sequences and db_name != 'uniprot':
      single_chain_msas.append(merged_msa)
      msa_size = len(set(merged_msa.sequences))
      print(f'{msa_size} unique sequences found in {db_name} for sequence {sequence_index}')
    elif merged_msa.sequences and db_name == 'uniprot':
      uniprot_msa = merged_msa

  notebook_utils.show_msa_info(single_chain_msas=single_chain_msas, sequence_index=sequence_index)

  # Turn the raw data into model features.
  feature_dict = {}
  feature_dict.update(pipeline.make_sequence_features(
      sequence=sequence, description='query', num_res=len(sequence)))
  feature_dict.update(pipeline.make_msa_features(msas=single_chain_msas))
  # We don't use templates in AlphaFold Colab notebook, add only empty placeholder features.
  feature_dict.update(notebook_utils.empty_placeholder_template_features(
      num_templates=0, num_res=len(sequence)))

  # Construct the all_seq features only for heteromers, not homomers.
  if model_type_to_use == notebook_utils.ModelType.MULTIMER and len(set(sequences)) > 1:
    valid_feats = msa_pairing.MSA_FEATURES + (
        'msa_uniprot_accession_identifiers',
        'msa_species_identifiers',
    )
    all_seq_features = {
        f'{k}_all_seq': v for k, v in pipeline.make_msa_features([uniprot_msa]).items()
        if k in valid_feats}
    feature_dict.update(all_seq_features)

  features_for_chain[protein.PDB_CHAIN_IDS[sequence_index - 1]] = feature_dict


# Do further feature post-processing depending on the model type.
if model_type_to_use == notebook_utils.ModelType.MONOMER:
  np_example = features_for_chain[protein.PDB_CHAIN_IDS[0]]

elif model_type_to_use == notebook_utils.ModelType.MULTIMER:
  all_chain_features = {}
  for chain_id, chain_features in features_for_chain.items():
    all_chain_features[chain_id] = pipeline_multimer.convert_monomer_features(
        chain_features, chain_id)

  all_chain_features = pipeline_multimer.add_assembly_features(all_chain_features)

  np_example = feature_processing.pair_and_merge(
      all_chain_features=all_chain_features, is_prokaryote=is_prokaryote)

  # Pad MSA to avoid zero-sized extra_msa.
  np_example = pipeline_multimer.pad_msa(np_example, min_num_seq=512)