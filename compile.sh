nvcc cuda_stock_simulator.cu -o cuda_sim 
g++ -fopenmp omp_stock_simulator.c -o omp_sim
g++ sequential_stock_simulator.cpp -o seq_sim

echo "finished compiling"