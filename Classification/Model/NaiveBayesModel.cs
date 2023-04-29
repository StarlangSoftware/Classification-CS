using System;
using System.Collections.Generic;
using System.IO;
using Classification.Attribute;
using Math;

namespace Classification.Model
{
    public class NaiveBayesModel : GaussianModel
    {
        private readonly Dictionary<string, Vector> _classMeans;
        private readonly Dictionary<string, Vector> _classDeviations;
        private readonly Dictionary<string, List<DiscreteDistribution>> _classAttributeDistributions;

        /**
         * <summary> A constructor that sets the priorDistribution, classMeans and classDeviations.</summary>
         *
         * <param name="priorDistribution">{@link DiscreteDistribution} input.</param>
         * <param name="classMeans">       A {@link HashMap} of String and {@link Vector}.</param>
         * <param name="classDeviations">  A {@link HashMap} of String and {@link Vector}.</param>
         */
        public NaiveBayesModel(DiscreteDistribution priorDistribution, Dictionary<string, Vector> classMeans,
            Dictionary<string, Vector> classDeviations)
        {
            this.priorDistribution = priorDistribution;
            _classMeans = classMeans;
            _classDeviations = classDeviations;
        }

        /**
         * <summary> A constructor that sets the priorDistribution and classAttributeDistributions.</summary>
         *
         * <param name="priorDistribution">          {@link DiscreteDistribution} input.</param>
         * <param name="classAttributeDistributions">{@link HashMap} of String and {@link ArrayList} of {@link DiscreteDistribution}s.</param>
         */
        public NaiveBayesModel(DiscreteDistribution priorDistribution,
            Dictionary<string, List<DiscreteDistribution>> classAttributeDistributions)
        {
            this.priorDistribution = priorDistribution;
            _classAttributeDistributions = classAttributeDistributions;
        }

        public NaiveBayesModel(string fileName)
        {
            var input = new StreamReader(fileName);
            var size = LoadPriorDistribution(input);
            _classMeans = LoadVectors(input, size);
            _classDeviations = LoadVectors(input, size);
            input.Close();
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
    }
}