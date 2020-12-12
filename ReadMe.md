## Localisation in Open quantum systems

The python file LvN_Solver.py can be run on a command line system(Terminal) with the following arguments.

user@SYSTEM:~/your_working_directory$  python3 LvN_Solver.py N (sys size) conf (ensemble size) T (LongTime) tSteps (no.of time steps) Method (type_of_disorder) env (bath_condition) rate (dephasing_rate) filename (name_for_your_output_file) 

N (sys size) - int()
The system size could be anything between 2-128 or even higher; N beyond 128 has not been tested. But, we expect to achieve reasonable amount of speed up with the object class implementation of Disorder_average_parallelisation():


conf (ensemble size) - int()
The no. of random configurations over which you average your populations. 


T (LongTime) - float() or int() works!
The Long time limit T upto which we propagate the density matrix before we carry out the time-average. 


tSteps (no.of time steps) - int()
No. of time steps involved in propagation of the rho from 0 to T

Method (type_of_disorder) str()type : 'site_disorder' or 'off_diagonal_disorder' 
The two arguments specify which Hamiltonian to select for the calculation.

env (bath_condition) str()type : 'pure_dephasing' or 'isolated'
Currently, we have two options to 'isolated' for the isolated quantum systems and 'pure_dephasing' to study the dephasing effect of the bath that kills coherences or off-diagonal elements of the density matrix in the site-energy basis

rate (dephasing_rate) : (float type)
Specify the dephasing rate for the system coupled to the bath. 
You need to set 0 for the case isolated system. 


filename (name_for_your_output_file) str() type :
This is for the file name for your data output, the code generate three files. First a Plot of long-time averages of disorder averaged populations, second txt file of the same data and finally, a .csv file with time dependent population data saved before long-time averages.
