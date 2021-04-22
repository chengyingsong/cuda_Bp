main:c_BP.cu Model.cu load_data.cu Config.cu Matrix.cu Matrix_cuda.cu 
	nvcc  Matrix_cuda.cu Matrix.cu c_BP.cu Model.cu load_data.cu Config.cu   -o main


.PHONY:clean
clean:
	-rm -rf *.o
	-rm -rf core