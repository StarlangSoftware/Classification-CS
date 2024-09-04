using System.Collections.Generic;
using System.IO;
using Classification.Attribute;
using Classification.InstanceList;
using Math;

namespace Classification.Model
{
    public class NaiveBayesModel : GaussianModel
    {
        private Dictionary<string, Vector> _classMeans;
        private Dictionary<string, Vector> _classDeviations;
        private Dictionary<string, List<DiscreteDistribution>> _classAttributeDistributions;

        public NaiveBayesModel()
        {
        }

        /// <summary>
        /// Loads a naive Bayes model from an input model file.
        /// </summary>
        /// <param name="fileName">Model file name.</param>
        private void Load(string fileName)
        {
            var input = new StreamReader(fileName);
            var size = LoadPriorDistribution(input);
            _classMeans = LoadVectors(input, size);
            _classDeviations = LoadVectors(input, size);
            input.Close();
        }

        /// <summary>
        /// Loads a naive Bayes model from an input model file.
        /// </summary>
        /// <param name="fileName">Model file name.</param>
        public NaiveBayesModel(string fileName)
        {
            Load(fileName);
        }

        /**
         * <summary> The calculateMetric method takes an {@link Instance} and a String as inputs and it returns the log likelihood of</summary>
         * these inputs.
         *
         * <param name="instance">{@link Instance} input.</param>
         * <param name="ci">      String input.</param>
         * <returns>The log likelihood of inputs.</returns>
         */
        protected override double CalculateMetric(Instance.Instance instance, string ci)
        {
            if (_classAttributeDistributions == null)
            {
                return LogLikelihoodContinuous(ci, instance);
            }

            return LogLikelihoodDiscrete(ci, instance);
        }

        /**
         * <summary> The logLikelihoodContinuous method takes an {@link Instance} and a class label as inputs. First it gets the logarithm</summary>
         * of given class label's probability via prior distribution as logLikelihood. Then it loops times of given instance attribute size, and accumulates the
         * logLikelihood by calculating -0.5 * ((xi - mi) / si )** 2).
         *
         * <param name="classLabel">String input class label.</param>
         * <param name="instance">  {@link Instance} input.</param>
         * <returns>The log likelihood of given class label and {@link Instance}.</returns>
         */
        private double LogLikelihoodContinuous(string classLabel, Instance.Instance instance)
        {
            var logLikelihood = System.Math.Log(priorDistribution.GetProbability(classLabel));
            for (var i = 0; i < instance.AttributeSize(); i++)
            {
                var xi = ((ContinuousAttribute) instance.GetAttribute(i)).GetValue();
                var mi = _classMeans[classLabel].GetValue(i);
                var si = _classDeviations[classLabel].GetValue(i);
                if (si != 0)
                {
                    logLikelihood += -0.5 * System.Math.Pow((xi - mi) / si, 2);
                }
            }

            return logLikelihood;
        }

        /**
         * <summary> The logLikelihoodDiscrete method takes an {@link Instance} and a class label as inputs. First it gets the logarithm</summary>
         * of given class label's probability via prior distribution as logLikelihood and gets the class attribute distribution of given class label.
         * Then it loops times of given instance attribute size, and accumulates the logLikelihood by calculating the logarithm of
         * corresponding attribute distribution's smoothed probability by using laplace smoothing on xi.
         *
         * <param name="classLabel">String input class label.</param>
         * <param name="instance">  {@link Instance} input.</param>
         * <returns>The log likelihood of given class label and {@link Instance}.</returns>
         */
        private double LogLikelihoodDiscrete(string classLabel, Instance.Instance instance)
        {
            var logLikelihood = System.Math.Log(priorDistribution.GetProbability(classLabel));
            var attributeDistributions = _classAttributeDistributions[classLabel];
            for (var i = 0; i < instance.AttributeSize(); i++)
            {
                var xi = ((DiscreteAttribute) instance.GetAttribute(i)).GetValue();
                logLikelihood += System.Math.Log(attributeDistributions[i].GetProbabilityLaplaceSmoothing(xi));
            }

            return logLikelihood;
        }
        
        /**
         * <summary> Training algorithm for Naive Bayes algorithm with a continuous data set.</summary>
         *
         * <param name="classLists">       Instances are divided into K lists, where each list contains only instances from a single class</param>
         */
        private void TrainContinuousVersion(Partition classLists)
        {
            _classMeans = new Dictionary<string, Vector>();
            _classDeviations = new Dictionary<string, Vector>();
            for (var i = 0; i < classLists.Size(); i++)
            {
                var classLabel = ((InstanceListOfSameClass) classLists.Get(i)).GetClassLabel();
                var averageVector = classLists.Get(i).Average().ToVector();
                _classMeans[classLabel] = averageVector;
                var standardDeviationVector = classLists.Get(i).StandardDeviation().ToVector();
                _classDeviations[classLabel] = standardDeviationVector;
            }

        }

        /**
         * <summary> Training algorithm for Naive Bayes algorithm with a discrete data set.</summary>
         * <param name="classLists">Instances are divided into K lists, where each list contains only instances from a single class</param>
         */
        private void TrainDiscreteVersion(Partition classLists)
        {
            _classAttributeDistributions =
                new Dictionary<string, List<DiscreteDistribution>>();
            for (var i = 0; i < classLists.Size(); i++)
            {
                _classAttributeDistributions[((InstanceListOfSameClass) classLists.Get(i)).GetClassLabel()] = 
                    classLists.Get(i).AllAttributesDistribution();
            }

        }

        /**
         * <summary> Training algorithm for Naive Bayes algorithm. It basically calls trainContinuousVersion for continuous data sets,
         * trainDiscreteVersion for discrete data sets.</summary>
         * <param name="trainSet">Training data given to the algorithm</param>
         * <param name="parameters">-</param>
         */
        public override void Train(InstanceList.InstanceList trainSet, Parameter.Parameter parameters)
        {
            priorDistribution = trainSet.ClassDistribution();
            var classLists = trainSet.DivideIntoClasses();
            if (classLists.Get(0).Get(0).GetAttribute(0) is DiscreteAttribute){
                TrainDiscreteVersion(classLists);
            } else {
                TrainContinuousVersion(classLists);
            }
        }

        /// <summary>
        /// Loads the naive Bayes model from an input file.
        /// </summary>
        /// <param name="fileName">File name of the naive Bayes model.</param>
        public override void LoadModel(string fileName)
        {
            Load(fileName);
        }

    }
}