using System.Collections.Generic;
using Math;

namespace Classification.Model
{
    public class QdaModel : LdaModel
    {
        private readonly Dictionary<string, Matrix> _W;

        /**
         * <summary> The constructor which sets the priorDistribution, w w1 and HashMap of String Matrix.</summary>
         *
         * <param name="priorDistribution">{@link DiscreteDistribution} input.</param>
         * <param name="W">                {@link HashMap} of String and Matrix.</param>
         * <param name="w">                {@link HashMap} of String and Vectors.</param>
         * <param name="w0">               {@link HashMap} of String and Double.</param>
         */
        public QdaModel(DiscreteDistribution priorDistribution, Dictionary<string, Matrix> W,
            Dictionary<string, Vector> w,
            Dictionary<string, double> w0) : base(priorDistribution, w, w0)
        {
            this._W = W;
        }

        /**
         * <summary> The calculateMetric method takes an {@link Instance} and a String as inputs. It multiplies Matrix Wi with Vector xi
         * then calculates the dot product of it with xi. Then, again it finds the dot product of wi and xi and returns the summation with w0i.</summary>
         *
         * <param name="instance">{@link Instance} input.</param>
         * <param name="ci">      String input.</param>
         * <returns>The result of Wi.multiplyWithVectorFromLeft(xi).dotProduct(xi) + wi.dotProduct(xi) + w0i.</returns>
         */
        protected override double CalculateMetric(Instance.Instance instance, string ci)
        {
            var xi = instance.ToVector();
            var Wi = _W[ci];
            var wi = w[ci];
            var w0i = w0[ci];
            return Wi.MultiplyWithVectorFromLeft(xi).DotProduct(xi) + wi.DotProduct(xi) + w0i;
        }
    }
}