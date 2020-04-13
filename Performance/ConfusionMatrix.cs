using System.Collections.Generic;
using DataStructure;

namespace Classification.Performance
{
    public class ConfusionMatrix
    {
        private readonly Dictionary<string, CounterHashMap<string>> _matrix;
        private readonly List<string> _classLabels;

        /**
         * <summary> Constructor that sets the class labels {@link List} and creates new {@link HashMap} matrix</summary>
         *
         * <param name="classLabels">{@link List} of string.</param>
         */
        public ConfusionMatrix(List<string> classLabels)
        {
            this._classLabels = classLabels;
            _matrix = new Dictionary<string, CounterHashMap<string>>();
        }

        /**
         * <summary> The classify method takes two strings; actual class and predicted class as inputs. If the matrix {@link HashMap} contains
         * given actual class string as a key, it then assigns the corresponding object of that key to a {@link CounterHashMap}, if not
         * it creates a new {@link CounterHashMap}. Then, it puts the given predicted class string to the counterHashMap and
         * also put this counterHashMap to the matrix {@link HashMap} together with the given actual class string.</summary>
         *
         * <param name="actualClass">   string input actual class.</param>
         * <param name="predictedClass">string input predicted class.</param>
         */
        public void Classify(string actualClass, string predictedClass)
        {
            CounterHashMap<string> counterHashMap;
            if (_matrix.ContainsKey(actualClass))
            {
                counterHashMap = _matrix[actualClass];
            }
            else
            {
                counterHashMap = new CounterHashMap<string>();
            }

            counterHashMap.Put(predictedClass);
            _matrix[actualClass] = counterHashMap;
        }

        /**
         * <summary> The addConfusionMatrix method takes a {@link ConfusionMatrix} as an input and loops through actual classes of that {@link HashMap}
         * and initially gets one row at a time. Then it puts the current row to the matrix {@link HashMap} together with the actual class string.</summary>
         *
         * <param name="confusionMatrix">{@link ConfusionMatrix} input.</param>
         */
        public void AddConfusionMatrix(ConfusionMatrix confusionMatrix)
        {
            foreach (var actualClass in confusionMatrix._matrix.Keys)
            {
                var rowToBeAdded = confusionMatrix._matrix[actualClass];
                if (_matrix.ContainsKey(actualClass))
                {
                    var currentRow = _matrix[actualClass];
                    currentRow.Add(rowToBeAdded);
                    _matrix[actualClass] = currentRow;
                }
                else
                {
                    _matrix[actualClass] = rowToBeAdded;
                }
            }
        }

        /**
         * <summary> The sumOfElements method loops through the keys in matrix {@link HashMap} and returns the summation of all the values of the keys.
         * I.e: TP+TN+FP+FN.</summary>
         *
         * <returns>The summation of values.</returns>
         */
        private double SumOfElements()
        {
            double result = 0;
            foreach (var actualClass in _matrix.Keys)
            {
                result += _matrix[actualClass].SumOfCounts();
            }

            return result;
        }

        /**
         * <summary> The trace method loops through the keys in matrix {@link HashMap} and if the current key contains the actual key,
         * it accumulates the corresponding values. I.e: TP+TN.</summary>
         *
         * <returns>Summation of values.</returns>
         */
        private double Trace()
        {
            double result = 0;
            foreach (var actualClass in _matrix.Keys)
            {
                if (_matrix[actualClass].ContainsKey(actualClass))
                {
                    result += _matrix[actualClass][actualClass];
                }
            }

            return result;
        }

        /**
         * <summary> The columnSum method takes a string predicted class as input, and loops through the keys in matrix {@link HashMap}.
         * If the current key contains the predicted class string, it accumulates the corresponding values. I.e: TP+FP.</summary>
         *
         * <param name="predictedClass">string input predicted class.</param>
         * <returns>Summation of values.</returns>
         */
        private double ColumnSum(string predictedClass)
        {
            double result = 0;
            foreach (var actualClass in _matrix.Keys)
            {
                if (_matrix[actualClass].ContainsKey(predictedClass))
                {
                    result += _matrix[actualClass][predictedClass];
                }
            }

            return result;
        }

        /**
         * <summary> The getAccuracy method returns the result of  TP+TN / TP+TN+FP+FN</summary>
         *
         * <returns>the result of  TP+TN / TP+TN+FP+FN</returns>
         */
        public double GetAccuracy()
        {
            return Trace() / SumOfElements();
        }

        /**
         * <summary> The precision method loops through the class labels and returns the resulting Array which has the result of TP/FP+TP.</summary>
         *
         * <returns>The result of TP/FP+TP.</returns>
         */
        public double[] Precision()
        {
            var result = new double[_classLabels.Count];
            for (var i = 0; i < _classLabels.Count; i++)
            {
                var actualClass = _classLabels[i];
                if (_matrix.ContainsKey(actualClass))
                {
                    result[i] = _matrix[actualClass][actualClass] / ColumnSum(actualClass);
                }
            }

            return result;
        }

        /**
         * <summary> The recall method loops through the class labels and returns the resulting Array which has the result of TP/FN+TP.</summary>
         *
         * <returns>The result of TP/FN+TP.</returns>
         */
        public double[] Recall()
        {
            var result = new double[_classLabels.Count];
            for (var i = 0; i < _classLabels.Count; i++)
            {
                var actualClass = _classLabels[i];
                if (_matrix.ContainsKey(actualClass))
                {
                    result[i] = (_matrix[actualClass][actualClass] + 0.0) /
                                _matrix[actualClass].SumOfCounts();
                }
            }

            return result;
        }

        /**
         * <summary> The fMeasure method loops through the class labels and returns the resulting Array which has the average of
         * recall and precision.</summary>
         *
         * <returns>The average of recall and precision.</returns>
         */
        public double[] FMeasure()
        {
            var precision = Precision();
            var recall = Recall();
            var result = new double[_classLabels.Count];
            for (var i = 0; i < _classLabels.Count; i++)
            {
                result[i] = 2 / (1 / precision[i] + 1 / recall[i]);
            }

            return result;
        }

        /**
         * <summary> The weightedFMeasure method loops through the class labels and returns the resulting Array which has the weighted average of
         * recall and precision.</summary>
         *
         * <returns>The weighted average of recall and precision.</returns>
         */
        public double WeightedFMeasure()
        {
            double[] fMeasure = FMeasure();
            double sum = 0;
            for (int i = 0; i < _classLabels.Count; i++)
            {
                string actualClass = _classLabels[i];
                sum += fMeasure[i] * _matrix[actualClass].SumOfCounts();
            }

            return sum / SumOfElements();
        }
    }
}