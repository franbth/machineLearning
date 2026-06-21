import algorithms.knn as knn
import algorithms.linear_regression_simple as linear_regression_simple
import algorithms.linear_regression_multiple as linear_regression_multiple
import algorithms.polynomial_regression as polynomial_regression
import algorithms.svr as svr

#algoritmos = {
#    "knn": knn.run
#}
#entrada = None
#while entrada != 0:
#    print("Ingrese: \n\t1.KNN\n\t0. Salir")
#    entrada = int(input())
#    match entrada:
#        case 1:
#            alg="knn"
#        case _:
#            break
    
#    algoritmos[alg]()

#knn.run("iris")
#linear_regression_simple.run("diabetes")
#linear_regression_multiple.run("automobile")
#polynomial_regression.run("synthetic")
svr.run("diabetes")

