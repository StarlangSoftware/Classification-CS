using System.Collections.Generic;
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