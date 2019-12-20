Installation
============

pip install -e .


Usage
=====

Key input is an array u_kj with the relative bias energy for each structure (with respect to each biased simulation/ window).

Here k indices structures and j biased simulations.

u_kj is structured as follows:

        Window index (stru_1), u_win_1(stru_1), .. u_win_n(stru_1) 
        Window index (stru_2), u_win_1(stru_2), .. u_win_n(stru_2) 
        ...
        Window index (stru_n), u_win_1(stru_n), .. u_win_n(stru_n)
        
Here u_win_n(stru_n), stands for the energy of structure n n in simulation window/run n. Each structure has to be evaluated under each simulation condition (i.e., the the energy function acting in each window). 

