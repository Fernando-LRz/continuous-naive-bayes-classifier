import pandas

class NaiveBayes:

    def __init__(self, training_dataset) -> None:       
        # Inicializar los atributos
        self.training_dataset = training_dataset
        self.class_name = training_dataset.columns[-1]
        self.attributes = training_dataset.columns[:-1]

        # Inicializar un diccionario para almacenar las estadísticas (media y desviación estándar) por atributo y clase
        self.statistics_by_class = {}

    def calculateStatisticsByClass(self) -> None:
        # Obtener una lista de las clases únicas en el conjunto de entrenamiento
        classes = self.training_dataset[self.class_name].unique()

        # Iterar sobre las clases
        for class_label in classes:
            # Filtrar el conjunto de entrenamiento para la clase actual
            class_data = self.training_dataset[self.training_dataset[self.class_name] == class_label]
            
            # Inicializar un diccionario para esta clase
            statistics_for_class = {}
            
            # Iterar sobre los atributos de la instancias del conjunto de entrenamiento
            for attribute_label in self.attributes: 
                # Calcular la media y desviación estándar para el atributo y clase actual
                mean = class_data[attribute_label].mean()
                std = class_data[attribute_label].std(ddof=0)
                
                # Almacenar las estadísticas en el diccionario para esta clase
                statistics_for_class[attribute_label] = {'media': mean, 'desviacion_estandar': std}
            
            # Almacenar el diccionario de estadísticas por clase
            self.statistics_by_class[class_label] = statistics_for_class
        
    def evaluate(self, test_dataset) -> pandas.DataFrame:
        # Crear una lista para almacenar las predicciones
        predictions = []

        # Crear una copia del conjunto de datos de prueba
        result = test_dataset.copy()

        # Iterar sobre el conjunto de datos de prueba
        for index, instance in test_dataset.iterrows():
            # Inicializar un diccionario para almacenar las probabilidades de clase
            class_probabilities = {}
            
            # Iterar sobre las clases
            for class_label, statistics_for_class in self.statistics_by_class.items():
                # Inicializar la probabilidad de clase con 1
                class_probability = 1.0

                print()
                print(class_label)
                print()
                print(statistics_for_class)
                
                # Iterar sobre los atributos de la instancia de prueba
                for attribute_label in self.attributes: 
                    # Obtener la media y desviación estándar del atributo para la clase actual
                    mean = statistics_for_class[attribute_label]['media']
                    std = statistics_for_class[attribute_label]['desviacion_estandar']
                    
                    # Calcular la probabilidad condicional para este atributo
                    attribute_value = instance[attribute_label]
                    conditional_probability = (1 / (std * (2 * 3.14159265359) ** 0.5)) * (2.71828182846 ** ((-0.5) * ((attribute_value - mean) / std) ** 2))
                    
                    # Multiplicar la probabilidad condicional por la probabilidad de clase
                    class_probability *= conditional_probability

                    print()
                    print('---------', attribute_label, '---------')
                    print()
                    print('media y desviacion_estandar')
                    print(mean, ' y ', std)
                    print()
                    print('valor en la instancia')
                    print(attribute_value)
                    print()
                    print('formula')
                    print(conditional_probability)
                

                print()
                print('---------', 'multiplicación', '---------')
                print()
                print(class_probability)
                # Almacenar la probabilidad de clase en el diccionario
                class_probabilities[class_label] = class_probability
            
            # Seleccionar la clase con la probabilidad más alta 
            predicted_class = max(class_probabilities, key=class_probabilities.get)

            # Guardar la predicción
            predictions.append(predicted_class)

        # Agregar la lista de predicciones como una nueva columna al dataframe
        result['Predicted_Class'] = predictions

        return result
    
    def getStatisticsByClass(self) -> dict:
        return self.statistics_by_class

    def fit(self) -> None:
        # Calcular las tablas de frecuencia
        self.calculateStatisticsByClass()

        # Calcular las tablas de verosimilitud
        # self.computeLikelihoodTables()