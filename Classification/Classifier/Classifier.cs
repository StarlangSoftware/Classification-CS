using System;
using System.Collections.Generic;
using Classification.Attribute;
using Classification.Performance;
using DataStructure;

namespace Classification.Classifier
{
    public abstract class Classifier
    {
        protected Model.Model model;

        public abstract void Train(InstanceList.InstanceList trainSet, Parameter.Parameter parameters);

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
                confusion.Classify(instance.GetClassLabel(), model.Predict(instance));
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
         * <summary> Accessor for the model.</summary>
         *
         * <returns>Model.</returns>
         */
        public Model.Model GetModel()
        {
            return model;
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
            foreach (var label in classLabels) {
                frequencies.Put(label);
            }
            return frequencies.Max();
        }
    }
}