using System;
using System.Collections.Generic;
using System.IO;
using Math;

namespace Classification.Model
{
    public class LdaModel : GaussianModel
    {
        protected Dictionary<string, double> w0;
        protected Dictionary<string, Vector> w;

        /**
         * <summary> A constructor which sets the priorDistribution, w and w0 according to given inputs.</summary>
         *
         * <param name="priorDistribution">{@link DiscreteDistribution} input.</param>
         * <param name="w">                {@link HashMap} of String and Vectors.</param>
         * <param name="w0">               {@link HashMap} of String and Double.</param>
         */
        public LdaModel(DiscreteDistribution priorDistribution, Dictionary<string, Vector> w,
            Dictionary<string, double> w0)
        {
            this.priorDistribution = priorDistribution;
            this.w = w;
            this.w0 = w0;
        }

        public LdaModel()
        {
            
        }
        
        /// <summary>
        /// Loads w0 and w hash maps from an input file. The number of items in the hash map is given by the parameter size.
        /// </summary>
        /// <param name="input">Input file</param>
        /// <param name="size">Number of items in the hash map read.</param>
        protected void LoadWandW0(StreamReader input, int size)
        {
            w0 = new Dictionary<string, double>();
            for (var i = 0; i < size; i++)
            {
                var line = input.ReadLine();
                var items = line.Split(" ");
                w0[items[0]] =  Double.Parse(items[1]);
            }

            w = LoadVectors(input, size);
        }

        /// <summary>
        /// Loads a Linear Discriminant Analysis model from an input model file.
        /// </summary>
        /// <param name="fileName">Model file name.</param>
        public LdaModel(string fileName)
        {
            var input = new StreamReader(fileName);
            var size = LoadPriorDistribution(input);
            LoadWandW0(input, size);
            input.Close();
        }

        /**
         * <summary> The calculateMetric method takes an {@link Instance} and a String as inputs. It returns the dot product of given Instance
         * and wi plus w0i.</summary>
         *
         * <param name="instance">{@link Instance} input.</param>
         * <param name="Ci">      String input.</param>
         * <returns>The dot product of given Instance and wi plus w0i.</returns>
         */
        protected override double CalculateMetric(Instance.Instance instance, string ci)
        {
            var xi = instance.ToVector();
            var wi = w[ci];
            var w0i = w0[ci];
            return wi.DotProduct(xi) + w0i;
        }
    }
}