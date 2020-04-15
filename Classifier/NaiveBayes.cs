using System.Collections.Generic;
using Classification.Attribute;
using Classification.InstanceList;
using Classification.Model;
using Math;

namespace Classification.Classifier
{
    public class NaiveBayes : Classifier
    {
        /**
         * <summary> Training algorithm for Naive Bayes algorithm with a continuous data set.</summary>
         *
         * <param name="priorDistribution">Probability distribution of classes P(C_i)</param>
         * <param name="classLists">       Instances are divided into K lists, where each list contains only instances from a single class</param>
         */
        private void TrainContinuousVersion(DiscreteDistribution priorDistribution, Partition classLists)
        {
            var classMeans = new Dictionary<string, Vector>();
            var classDeviations = new Dictionary<string, Vector>();
            for (var i = 0; i < classLists.Size(); i++)
            {
                var classLabel = ((InstanceListOfSameClass) classLists.Get(i)).GetClassLabel();
                var averageVector = classLists.Get(i).Average().ToVector();
                classMeans[classLabel] = averageVector;
                var standardDeviationVector = classLists.Get(i).StandardDeviation().ToVector();
                classDeviations[classLabel] = standardDeviationVector;
            }

            model = new NaiveBayesModel(priorDistribution, classMeans, classDeviations);
        }

        /**
         * <summary> Training algorithm for Naive Bayes algorithm with a discrete data set.</summary>
         * <param name="priorDistribution">Probability distribution of classes P(C_i)</param>
         * <param name="classLists">Instances are divided into K lists, where each list contains only instances from a single class</param>
         */
        private void TrainDiscreteVersion(DiscreteDistribution priorDistribution, Partition classLists)
        {
            var classAttributeDistributions =
                new Dictionary<string, List<DiscreteDistribution>>();
            for (var i = 0; i < classLists.Size(); i++)
            {
                classAttributeDistributions[((InstanceListOfSameClass) classLists.Get(i)).GetClassLabel()] = 
                    classLists.Get(i).AllAttributesDistribution();
            }

            model = new NaiveBayesModel(priorDistribution, classAttributeDistributions);
        }

        /**
         * <summary> Training algorithm for Naive Bayes algorithm. It basically calls trainContinuousVersion for continuous data sets,
         * trainDiscreteVersion for discrete data sets.</summary>
         * <param name="trainSet">Training data given to the algorithm</param>
         * <param name="parameters">-</param>
         */
        public override void Train(InstanceList.InstanceList trainSet, Parameter.Parameter parameters)
        {
            var priorDistribution = trainSet.ClassDistribution();
            var classLists = trainSet.DivideIntoClasses();
            if (classLists.Get(0).Get(0).GetAttribute(0) is DiscreteAttribute){
                TrainDiscreteVersion(priorDistribution, classLists);
            } else {
                TrainContinuousVersion(priorDistribution, classLists);
            }
        }
    }
}