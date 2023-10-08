import pandas
from NaiveBayes import NaiveBayes

def main():
    # Archivo con el set de datos
    csv = '../iris.csv'

    # Leer el archivo CSV y almacenar los datos en un DataFrame
    dataset = pandas.read_csv(csv)

    # Eliminar los espacios en blanco del DataFrame
    dataset = dataset.map(lambda x: x.strip() if isinstance(x, str) else x)

    # Definir el tamaño del set de datos de entrenamiento
    training_percentage = 0.7
    number_of_instances = round(len(dataset) * training_percentage)
    
    # Definir el set de datos de entrenamiento, seleccionando aleatoriamente las instancias
    training_dataset = dataset.sample(number_of_instances)

    # Definir el set de datos de prueba
    test_dataset = dataset.drop(training_dataset.index)

    # Crear una instancia de la clase NaiveBayes
    naiveBayes = NaiveBayes(dataset)

    naiveBayes.fit()
    result = naiveBayes.evaluate(test_dataset)
    
    print(result)

# Ejecutar el main
if __name__ == '__main__':
    main()   