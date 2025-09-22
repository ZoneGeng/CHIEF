import torch
import numpy as np



def featurize(batch, device):
    alphabet = 'ACDEFGHIKLMNPQRSTVWYX'
    B = len(batch)
    lengths = np.array([len(b['seq']) for b in batch], dtype=np.int32) #sum of chain seq lengths
    L_max = max([len(b['seq']) for b in batch])
    X = np.zeros([B, L_max, 4, 3])
    residue_idx = -100*np.ones([B, L_max], dtype=np.int32) #residue idx with jumps across chains
    chain_M = np.zeros([B, L_max], dtype=np.int32) #1.0 for the bits that need to be predicted, 0.0 for the bits that are given
    mask_self = np.ones([B, L_max, L_max], dtype=np.int32) #for interface loss calculation - 0.0 for self interaction, 1.0 for other
    chain_encoding_all = np.zeros([B, L_max], dtype=np.int32) #integer encoding for chains 0, 0, 0,...0, 1, 1,..., 1, 2, 2, 2...
    S = np.zeros([B, L_max], dtype=np.int32) #sequence AAs integers
    init_alphabet = ['A', 'B', 'C', 'D', 'E', 'F', 'G','H', 'I', 'J','K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T','U', 'V','W','X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g','h', 'i', 'j','k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't','u', 'v','w','x', 'y', 'z']
    extra_alphabet = [str(item) for item in list(np.arange(300))]
    chain_letters = init_alphabet + extra_alphabet
    for i, b in enumerate(batch):
        masked_chains = b['masked_list']
        visible_chains = b['visible_list']
        all_chains = masked_chains + visible_chains
        visible_temp_dict = {}
        masked_temp_dict = {}
        for step, letter in enumerate(all_chains):
            chain_seq = b[f'seq_chain_{letter}']
            if letter in visible_chains:
                visible_temp_dict[letter] = chain_seq
            elif letter in masked_chains:
                masked_temp_dict[letter] = chain_seq
        for km, vm in masked_temp_dict.items():
            for kv, vv in visible_temp_dict.items():
                if vm == vv:
                    if kv not in masked_chains:
                        masked_chains.append(kv)
                    if kv in visible_chains:
                        visible_chains.remove(kv)
        all_chains = masked_chains + visible_chains
        # random.shuffle(all_chains) #randomly shuffle chain order
        num_chains = b['num_of_chains']
        mask_dict = {}
        x_chain_list = []
        chain_mask_list = []
        chain_seq_list = []
        chain_encoding_list = []
        c = 1
        l0 = 0
        l1 = 0
        for step, letter in enumerate(all_chains):
            if letter in visible_chains:
                chain_seq = b[f'seq_chain_{letter}']
                chain_length = len(chain_seq)
                chain_coords = b[f'coords_chain_{letter}'] #this is a dictionary
                chain_mask = np.zeros(chain_length) #0.0 for visible chains
                x_chain = np.stack([chain_coords[c] for c in [f'N_chain_{letter}', f'CA_chain_{letter}', f'C_chain_{letter}', f'O_chain_{letter}']], 1) #[chain_length,4,3]
                x_chain_list.append(x_chain)
                chain_mask_list.append(chain_mask)
                chain_seq_list.append(chain_seq)
                chain_encoding_list.append(c*np.ones(np.array(chain_mask).shape[0]))
                l1 += chain_length
                mask_self[i, l0:l1, l0:l1] = np.zeros([chain_length, chain_length])
                residue_idx[i, l0:l1] = 100*(c-1)+np.arange(l0, l1)
                l0 += chain_length
                c+=1
            elif letter in masked_chains: 
                chain_seq = b[f'seq_chain_{letter}']
                chain_length = len(chain_seq)
                chain_coords = b[f'coords_chain_{letter}'] #this is a dictionary
                chain_mask = np.ones(chain_length) #0.0 for visible chains
                x_chain = np.stack([chain_coords[c] for c in [f'N_chain_{letter}', f'CA_chain_{letter}', f'C_chain_{letter}', f'O_chain_{letter}']], 1) #[chain_lenght,4,3]
                x_chain_list.append(x_chain)
                chain_mask_list.append(chain_mask)
                chain_seq_list.append(chain_seq)
                chain_encoding_list.append(c*np.ones(np.array(chain_mask).shape[0]))
                l1 += chain_length
                mask_self[i, l0:l1, l0:l1] = np.zeros([chain_length, chain_length])
                residue_idx[i, l0:l1] = 100*(c-1)+np.arange(l0, l1)
                l0 += chain_length
                c+=1
        x = np.concatenate(x_chain_list,0) #[L, 4, 3]
        all_sequence = "".join(chain_seq_list)
        m = np.concatenate(chain_mask_list,0) #[L,], 1.0 for places that need to be predicted
        chain_encoding = np.concatenate(chain_encoding_list,0)

        l = len(all_sequence)
        x_pad = np.pad(x, [[0,L_max-l], [0,0], [0,0]], 'constant', constant_values=(np.nan, ))
        X[i,:,:,:] = x_pad

        m_pad = np.pad(m, [[0,L_max-l]], 'constant', constant_values=(0.0, ))
        chain_M[i,:] = m_pad

        chain_encoding_pad = np.pad(chain_encoding, [[0,L_max-l]], 'constant', constant_values=(0.0, ))
        chain_encoding_all[i,:] = chain_encoding_pad

        # Convert to labels
        indices = np.asarray([alphabet.index(a) if a != '-' else alphabet.index('X') for a in all_sequence], dtype=np.int32)
        S[i, :l] = indices

    isnan = np.isnan(X)
    mask = np.isfinite(np.sum(X,(2,3))).astype(np.float32) # 1.0 is for finite, 0.0 is for missing value
    X[isnan] = 0.

    # Conversion
    residue_idx = torch.from_numpy(residue_idx).to(dtype=torch.long,device=device)
    S = torch.from_numpy(S).to(dtype=torch.long,device=device)
    X = torch.from_numpy(X).to(dtype=torch.float32, device=device)
    mask = torch.from_numpy(mask).to(dtype=torch.float32, device=device)
    mask_self = torch.from_numpy(mask_self).to(dtype=torch.float32, device=device)
    chain_M = torch.from_numpy(chain_M).to(dtype=torch.float32, device=device)
    chain_encoding_all = torch.from_numpy(chain_encoding_all).to(dtype=torch.long, device=device)
    return X, S, mask, lengths, chain_M, residue_idx, mask_self, chain_encoding_all


def PDBparser(folder_with_pdbs_path,chain='single'):
    folder_with_pdbs_path = folder_with_pdbs_path.rstrip('/')
    import numpy as np
    import os, time, gzip, json
    import glob 
    
    # folder_with_pdbs_path = args.input_path
    # save_path = args.output_path
    ca_only = False
    
    alpha_1 = list("ARNDCQEGHILKMFPSTWYV-")
    states = len(alpha_1)
    alpha_3 = ['ALA','ARG','ASN','ASP','CYS','GLN','GLU','GLY','HIS','ILE',
               'LEU','LYS','MET','PHE','PRO','SER','THR','TRP','TYR','VAL','GAP']
    
    aa_1_N = {a:n for n,a in enumerate(alpha_1)}
    aa_3_N = {a:n for n,a in enumerate(alpha_3)}
    aa_N_1 = {n:a for n,a in enumerate(alpha_1)}
    aa_1_3 = {a:b for a,b in zip(alpha_1,alpha_3)}
    aa_3_1 = {b:a for a,b in zip(alpha_1,alpha_3)}
    
    def AA_to_N(x):
      # ["ARND"] -> [[0,1,2,3]]
      x = np.array(x);
      if x.ndim == 0: x = x[None]
      return [[aa_1_N.get(a, states-1) for a in y] for y in x]
    
    def N_to_AA(x):
      # [[0,1,2,3]] -> ["ARND"]
      x = np.array(x);
      if x.ndim == 1: x = x[None]
      return ["".join([aa_N_1.get(a,"-") for a in y]) for y in x]
    
    
    def parse_PDB_biounits(x, atoms=['N','CA','C'], chain=None):
      '''
      input:  x = PDB filename
              atoms = atoms to extract (optional)
      output: (length, atoms, coords=(x,y,z)), sequence
      '''
      xyz,seq,min_resn,max_resn = {},{},1e6,-1e6
      for line in open(x,"rb"):
        line = line.decode("utf-8","ignore").rstrip()
    
        if line[:6] == "HETATM" and line[17:17+3] == "MSE":
          line = line.replace("HETATM","ATOM  ")
          line = line.replace("MSE","MET")
    
        if line[:4] == "ATOM":
          ch = line[21:22]
          if ch == chain or chain is None:
            atom = line[12:12+4].strip()
            resi = line[17:17+3]
            resn = line[22:22+5].strip()
            x,y,z = [float(line[i:(i+8)]) for i in [30,38,46]]
    
            if resn[-1].isalpha(): 
                resa,resn = resn[-1],int(resn[:-1])-1
            else: 
                resa,resn = "",int(resn)-1
    #         resn = int(resn)
            if resn < min_resn: 
                min_resn = resn
            if resn > max_resn: 
                max_resn = resn
            if resn not in xyz: 
                xyz[resn] = {}
            if resa not in xyz[resn]: 
                xyz[resn][resa] = {}
            if resn not in seq: 
                seq[resn] = {}
            if resa not in seq[resn]: 
                seq[resn][resa] = resi
    
            if atom not in xyz[resn][resa]:
              xyz[resn][resa][atom] = np.array([x,y,z])
    
      # convert to numpy arrays, fill in missing values
      seq_,xyz_ = [],[]
      try:
          for resn in range(min_resn,max_resn+1):
            if resn in seq:
              for k in sorted(seq[resn]): seq_.append(aa_3_N.get(seq[resn][k],20))
            else: seq_.append(20)
            if resn in xyz:
              for k in sorted(xyz[resn]):
                for atom in atoms:
                  if atom in xyz[resn][k]: xyz_.append(xyz[resn][k][atom])
                  else: xyz_.append(np.full(3,np.nan))
            else:
              for atom in atoms: xyz_.append(np.full(3,np.nan))
          return np.array(xyz_).reshape(-1,len(atoms),3), N_to_AA(np.array(seq_))
      except TypeError:
          return 'no_chain', 'no_chain'
    
    
    
    pdb_dict_list = []
    c = 0
    
    if os.path.isfile(folder_with_pdbs_path) and folder_with_pdbs_path.endswith('.pdb'):
        biounit_names = [folder_with_pdbs_path]
        
    elif os.path.isdir(folder_with_pdbs_path):
        if folder_with_pdbs_path[-1]!='/':
            folder_with_pdbs_path = folder_with_pdbs_path+'/'
            biounit_names = glob.glob(folder_with_pdbs_path+'*.pdb')
    
    
    init_alphabet = ['A', 'B', 'C', 'D', 'E', 'F', 'G','H', 'I', 'J','K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T','U', 'V','W','X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g','h', 'i', 'j','k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't','u', 'v','w','x', 'y', 'z']
    extra_alphabet = [str(item) for item in list(np.arange(300))]
    chain_alphabet = init_alphabet + extra_alphabet
    
    for biounit in biounit_names:
        my_dict = {}
        s = 0
        chains_list = []
        concat_seq = ''
        concat_N = []
        concat_CA = []
        concat_C = []
        concat_O = []
        concat_mask = []
        coords_dict = {}
        for letter in chain_alphabet:
            if ca_only:
                sidechain_atoms = ['CA']
            else:
                sidechain_atoms = ['N', 'CA', 'C', 'O']
            xyz, seq = parse_PDB_biounits(biounit, atoms=sidechain_atoms, chain=letter)
            if type(xyz) != str:
                concat_seq += seq[0]
                my_dict['seq_chain_'+letter]=seq[0]
                coords_dict_chain = {}
                if ca_only:
                    coords_dict_chain['CA_chain_'+letter]=xyz.tolist()
                else:
                    coords_dict_chain['N_chain_' + letter] = xyz[:, 0, :].tolist()
                    coords_dict_chain['CA_chain_' + letter] = xyz[:, 1, :].tolist()
                    coords_dict_chain['C_chain_' + letter] = xyz[:, 2, :].tolist()
                    coords_dict_chain['O_chain_' + letter] = xyz[:, 3, :].tolist()
                my_dict['coords_chain_'+letter]=coords_dict_chain
                s += 1
                chains_list.append(letter)
        fi = biounit.rfind("/")
        my_dict['name']=biounit[(fi+1):-4]
        my_dict['num_of_chains'] = s
        my_dict['seq'] = concat_seq
        my_dict['chains_list'] = chains_list
        
        if chain == 'single':
            if len(my_dict['chains_list']) == 1:
                my_dict['masked_list'] = my_dict['chains_list']
                my_dict['visible_list'] = []
            else:
                continue
            
        if chain == 'multiple':
            if len(my_dict['chains_list']) > 1:
                my_dict['masked_list'] = my_dict['chains_list']
                my_dict['visible_list'] = []
            else:
                continue
        
        if chain == 'all':
            my_dict['masked_list'] = my_dict['chains_list']
            my_dict['visible_list'] = []
            
        if s < len(chain_alphabet):
            pdb_dict_list.append(my_dict)
            c+=1
        
    return pdb_dict_list