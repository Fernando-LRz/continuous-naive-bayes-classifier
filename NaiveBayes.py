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
                statistics_for_class[attribute_label] = {'mean': mean, 'standard_deviation': std}
            
            # Almacenar el diccionario de estadísticas por clase
            self.statistics_by_class[class_label] = statistics_for_class
        
    def evaluate(self, test_dataset) -> pandas.DataFrame:
        # Crear una lista para almacenar las predicciones
        predictions = []

        # Crear una copia del conjunto de datos de prueba
        result = test_dataset.copy()

        # Obtener las etiquetas de clase únicas del conjunto de entrenamiento
        unique_class_labels = self.training_dataset[self.class_name].unique()

        # Calcular las probabilidades a priori de las clases en el conjunto de entrenamiento
        class_counts = self.training_dataset[self.class_name].value_counts()
        total_instances = len(self.training_dataset)
        prior_probabilities = {class_label: class_counts[class_label] / total_instances for class_label in unique_class_labels}

        # Iterar sobre el conjunto de datos de prueba
        for index, instance in test_dataset.iterrows():
            # Inicializar un diccionario para almacenar las probabilidades de clase
            class_probabilities = {}
            
            # Iterar sobre las clases
            for class_label, statistics_for_class in self.statistics_by_class.items():
                # Inicializar la probabilidad de clase con 1
                class_probability = 1.0

                '''
                print()
                print(class_label)
                print()
                print(statistics_for_class) 
                '''
                
                # Iterar sobre los atributos de la instancia de prueba
                for attribute_label in self.attributes: 
                    # Obtener la media y desviación estándar del atributo para la clase actual
                    mean = statistics_for_class[attribute_label]['mean']
                    std = statistics_for_class[attribute_label]['standard_deviation']
                    
                    # Calcular la probabilidad condicional para este atributo
                    attribute_value = instance[attribute_label]
                    attribute_probability = (1 / (std * (2 * 3.14159265359) ** 0.5)) * (2.71828182846 ** ((-0.5) * ((attribute_value - mean) / std) ** 2))
                    
                    # Multiplicar la probabilidad condicional por la probabilidad de clase
                    class_probability *= attribute_probability

                    '''
                    print()
                    print('---------', attribute_label, '---------')
                    print()
                    print('media y desviación estandar')
                    print(mean, ' - ', std)
                    print()
                    print('valor en la instancia')
                    print(attribute_value)
                    print()
                    print('formula')
                    print(attribute_probability)
                    '''

                # Multiplicar la probabilidad a priori de la clase
                class_probability *= prior_probabilities[class_label]

                # Almacenar la probabilidad de clase en el diccionario
                class_probabilities[class_label] = class_probability

                '''
                print()
                print('---------', 'probabilidad de la clase', '---------')
                print()
                print(class_probability)
                print('----------------------------------------')
                '''
            
            # Seleccionar la clase con la probabilidad más alta 
            predicted_class = max(class_probabilities, key=class_probabilities.get)

            # Guardar la predicción
            predictions.append(predicted_class)

            '''
            print()
            print('---------', 'predicción', '---------')
            print()
            print(class_probabilities)
            print()
            print(predicted_class)
            print('------------------------------------')
            '''

        # Agregar la lista de predicciones como una nueva columna al dataframe
        result['predicted_class'] = predictions

        # Comparar la clase esperada y la clase predecida
        result['match'] = result[self.class_name] == result['predicted_class']

        return result
    
    def computeConfusionMatrix(self, result):
        # Crear un dataframe para la matriz de confusión
        confusion_matrix = pandas.crosstab(result[self.class_name], result['predicted_class'], rownames=['Actual'], colnames=['Predicted'])

        # Calcular la precisión para cada clase
        class_precisions = confusion_matrix.apply(lambda col: col[col.name] / col.sum(), axis=0)

        # Calcular el recall para cada clase
        class_recalls = confusion_matrix.apply(lambda row: row[row.name] / row.sum(), axis=1)

        # Calcular la exactitud del modelo
        correct_predictions = sum(confusion_matrix[i][i] for i in confusion_matrix.index)
        total_predictions = confusion_matrix.sum().sum()
        accuracy = correct_predictions / total_predictions

        # Agregar una fila para el recall y una columna para la precisión
        confusion_matrix.loc['Recall'] = class_recalls
        confusion_matrix['Precision'] = class_precisions

        return confusion_matrix, accuracy
    
    def getStatisticsByClass(self) -> dict:
        return self.statistics_by_class

    def fit(self) -> None:
        # Calcular las tablas de frecuencia
        self.calculateStatisticsByClass()