using System;
using System.Collections.Generic;
using System.IO;
using Classification.Attribute;
using Classification.Performance;
using DataStructure;
using Math;

namespace Classification.Model
{
    public abstract class Model
    {
        public abstract void Train(InstanceList.InstanceList trainSet, Parameter.Parameter parameters);

        public abstract void LoadModel(string fileName);

        /**
         * <summary> Checks given instance's attribute and returns true if it is a discrete indexed attribute, false otherwise.</summary>
         *
         * <param name="instance">Instance to check.</param>
         * <returns>True if instance is a discrete indexed attribute, false otherwise.</returns>
         */
        public bool DiscreteCheck(Instance.Instance instance)
        {
            for (var i = 0; i < instance.AttributeSize(); i++)
            {
                if (instance.GetAttribute(i) is DiscreteAttribute && !(instance.GetAttribute(i) is
                        DiscreteIndexedAttribute))
                {
                    return false;
                }
            }

            return true;
        }

        /**
         * <summary> TestClassification an instance list with the current model.</summary>
         *
         * <param name="testSet">Test data (list of instances) to be tested.</param>
         * <returns>The accuracy (and error) of the model as an instance of Performance class.</returns>
         */
        public virtual Performance.Performance Test(InstanceList.InstanceList testSet)
        {
            var classLabels = testSet.GetUnionOfPossibleClassLabels();
            var confusion = new ConfusionMatrix(classLabels);
            for (var i = 0; i < testSet.Size(); i++)
            {
                var instance = testSet.Get(i);
                confusion.Classify(instance.GetClassLabel(), Predict(instance));
            }

            return new DetailedClassificationPerformance(confusion);
        }

        /**
         * <summary> Runs current classifier with the given train and test data.</summary>
         *
         * <param name="parameter">Parameter of the classifier to be trained.</param>
         * <param name="trainSet"> Training data to be used in training the classifier.</param>
         * <param name="testSet">  Test data to be tested after training the model.</param>
         * <returns>The accuracy (and error) of the trained model as an instance of Performance class.</returns>
         */
        public Performance.Performance SingleRun(Parameter.Parameter parameter, InstanceList.InstanceList trainSet,
            InstanceList.InstanceList testSet)
        {
            Train(trainSet, parameter);
            return Test(testSet);
        }


        /**
         * <summary> An abstract predict method that takes an {@link Instance} as an input.</summary>
         *
         * <param name="instance">{@link Instance} to make prediction.</param>
         * <returns>The class label as a String.</returns>
         */
        public abstract string Predict(Instance.Instance instance);

        public abstract Dictionary<string, double> PredictProbability(Instance.Instance instance);

        /// <summary>
        /// Loads an instance list from an input model file.
        /// </summary>
        /// <param name="input">Input model file.</param>
        /// <returns>Instance list read from an input model file.</returns>
        protected InstanceList.InstanceList LoadInstanceList(StreamReader input)
        {
            var types = input.ReadLine().Split(" ");
            var instanceCount = int.Parse(input.ReadLine());
            var instanceList = new InstanceList.InstanceList();
            for (var i = 0; i < instanceCount; i++)
            {
                instanceList.Add(LoadInstance(input.ReadLine(), types));
            }

            return instanceList;
        }

        /// <summary>
        /// Loads a discrete distribution from an input model file
        /// </summary>
        /// <param name="input">Input model file.</param>
        /// <returns>Discrete distribution read from an input model file.</returns>
        public static DiscreteDistribution LoadDiscreteDistribution(StreamReader input)
        {
            var distribution = new DiscreteDistribution();
            var size = int.Parse(input.ReadLine());
            for (var i = 0; i < size; i++)
            {
                var line = input.ReadLine();
                var items = line.Split(" ");
                var count = int.Parse(items[1]);
                for (var j = 0; j < count; j++)
                {
                    distribution.AddItem(items[0]);
                }
            }

            return distribution;
        }

        /// <summary>
        /// Loads a single instance from a single line.
        /// </summary>
        /// <param name="line">Line containing the instance.</param>
        /// <param name="attributeTypes">Type of the attributes of the instance. If th attribute is discrete, it is
        /// "DISCRETE", otherwise it is "CONTINUOUS".</param>
        /// <returns></returns>
        protected Instance.Instance LoadInstance(string line, string[] attributeTypes)
        {
            var items = line.Split(",");
            var instance = new Instance.Instance(items[items.Length - 1]);
            for (var i = 0; i < items.Length - 1; i++)
            {
                switch (attributeTypes[i])
                {
                    case "DISCRETE":
                        instance.AddAttribute(items[i]);
                        break;
                    case "CONTINUOUS":
                        instance.AddAttribute(double.Parse(items[i]));
                        break;
                }
            }

            return instance;
        }

        /// <summary>
        /// Loads a matrix from an input model file.
        /// </summary>
        /// <param name="input">Input model file.</param>
        /// <returns>Matrix read from the input model file.</returns>
        protected Matrix LoadMatrix(StreamReader input)
        {
            var items = input.ReadLine().Split(" ");
            var matrix = new Matrix(int.Parse(items[0]), int.Parse(items[1]));
            for (var j = 0; j < matrix.GetRow(); j++)
            {
                var line = input.ReadLine();
                items = line.Split(" ");
                for (var k = 0; k < matrix.GetColumn(); k++)
                {
                    matrix.SetValue(j, k, Double.Parse(items[k]));
                }
            }

            return matrix;
        }

        /**
         * <summary> Given an array of class labels, returns the maximum occurred one.</summary>
         *
         * <param name="classLabels">An array of class labels.</param>
         * <returns>The class label that occurs most in the array of class labels (mod of class label list).</returns>
         */
        public static string GetMaximum(List<string> classLabels)
        {
            var frequencies = new CounterHashMap<string>();
            foreach (var label in classLabels)
            {
                frequencies.Put(label);
            }

            return frequencies.Max();
        }
    }
}