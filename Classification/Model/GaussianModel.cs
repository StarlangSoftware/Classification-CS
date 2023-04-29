using System.Collections.Generic;
using System.IO;
using Classification.Instance;
using Math;

namespace Classification.Model
{
    public abstract class GaussianModel : ValidatedModel
    {
        protected DiscreteDistribution priorDistribution;

        /**
         * <summary> Abstract method calculateMetric takes an {@link Instance} and a string as inputs.</summary>
         *
         * <param name="instance">{@link Instance} input.</param>
         * <param name="Ci">      string input.</param>
         * <returns>A double value as metric.</returns>
         */
        protected abstract double CalculateMetric(Instance.Instance instance, string Ci);

        protected int LoadPriorDistribution(StreamReader input)
        {
            var size = int.Parse(input.ReadLine());
            priorDistribution = new DiscreteDistribution();
            for (var i = 0; i < size; i++)
            {
                var line = input.ReadLine();
                var items = line.Split(" ");
                for (var j = 0; j < int.Parse(items[1]); j++)
                {
                    priorDistribution.AddItem(items[0]);
                }
            }

            return size;
        }

        protected Dictionary<string, Vector> LoadVectors(StreamReader input, int size)
        {
            var map = new Dictionary<string, Vector>();
            for (var i = 0; i < size; i++)
            {
                var line = input.ReadLine();
                var items = line.Split(" ");
                var vector = new Vector(int.Parse(items[1]), 0);
                for (var j = 2; j < items.Length; j++)
                {
                    vector.SetValue(j - 2, double.Parse(items[j]));
                }

                map[items[0]] = vector;
            }

            return map;
        }

        /**
         * <summary> The predict method takes an Instance as an input. First it gets the size of prior distribution and loops this size times.
         * Then it gets the possible class labels and and calculates metric value. At the end, it returns the class which has the
         * maximum value of metric.</summary>
         *
         * <param name="instance">{@link Instance} to predict.</param>
         * <returns>The class which has the maximum value of metric.</returns>
         */
        public override string Predict(Instance.Instance instance)
        {
            string predictedClass;
            var maxMetric = double.MinValue;
            int size;
            if (instance is CompositeInstance compositeInstance)
            {
                predictedClass = compositeInstance.GetPossibleClassLabels()[0];
                size = compositeInstance.GetPossibleClassLabels().Count;
            }
            else
            {
                predictedClass = priorDistribution.GetMaxItem();
                size = priorDistribution.Count;
            }

            for (var i = 0; i < size; i++)
            {
                string ci;
                if (instance is CompositeInstance compositeInstance1)
                {
                    ci = compositeInstance1.GetPossibleClassLabels()[i];
                }
                else
                {
                    ci = priorDistribution.GetItem(i);
                }

                if (priorDistribution.ContainsItem(ci))
                {
                    var metric = CalculateMetric(instance, ci);
                    if (metric > maxMetric)
                    {
                        maxMetric = metric;
                        predictedClass = ci;
                    }
                }
            }

            return predictedClass;
        }

        public override Dictionary<string, double> PredictProbability(Instance.Instance instance)
        {
            return null;
        }
    }
}