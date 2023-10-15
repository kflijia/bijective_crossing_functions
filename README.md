 # --- Demo of Causal Representation Learning (CRL) ---
    by Jia Li jiaxx213@umn.edu 
    project finished in Aug 2022, published Feb 2023

This folder includes all python source code, and some log files.
Please read the Jupyter Notebook demo_CRL_scripts.ipynb for all introductions to this demo.
Rerunning demo_CRL_scripts.ipynb needs the source data or the intermediate results data, please download the full package: 

https://drive.google.com/file/d/1BctiEa41ZofheBrH0hYIwhpPZkmotmro/view?usp=sharing


# --- Important Update! The New Name is Relation-Indexed Representation Learning (RIRL)  ---
    by Jia Li jiaxx213@umn.edu  Oct 2023

I realized how silly it was for me to use RNN-GRU in this work, which fixed the Effect Time Window T_y = 1
Since during this experiment, I haven't recognized the role of Temporal Nonlinear Distribution, AKA dynamics, yet

Please directly use time sequences as the inputs, for both Tower A (the cause) and Tower B (the effect), and forget about GRU

Please refer to the paper ->      Relation-Oriented: Toward Causal Knowledge-Aligned AGI 
