using System;
using Classification.Attribute;

namespace Classification.DistanceMetric
{
    public class EuclidianDistance : DistanceMetric
    {

        /**
         * <summary> Calculates Euclidian distance between two instances. For continuous features: \sum_{i=1}^d (x_i^(1) - x_i^(2))^2,
         * For discrete features: \sum_{i=1}^d 1(x_i^(1) == x_i^(2))</summary>
         *
         * <param name="instance1">First instance</param>
         * <param name="instance2">Second instance</param>
         * <returns>Euclidian distance between two instances.</returns>
         */
        public double Distance(Instance.Instance instance1, Instance.Instance instance2)
        {
            double result = 0;
            for (var i = 0; i < instance1.AttributeSize(); i++)
            {
                if (instance1.GetAttribute(i) is DiscreteAttribute && instance2.GetAttribute(i) is DiscreteAttribute)
                {
                    if (((DiscreteAttribute) instance1.GetAttribute(i)).GetValue() != null &&
                        string.Compare(((DiscreteAttribute) instance1.GetAttribute(i)).GetValue(),
                            ((DiscreteAttribute) instance2.GetAttribute(i)).GetValue(), StringComparison.Ordinal) != 0)
                    {
                        result += 1;
                    }
                }
                else
                {
                    if (instance1.GetAttribute(i) is ContinuousAttribute && instance2.GetAttribute(i) is
                        ContinuousAttribute)
                    {
                        result += System.Math.Pow(
                            ((ContinuousAttribute) instance1.GetAttribute(i)).GetValue() -
                            ((ContinuousAttribute) instance2.GetAttribute(i)).GetValue(), 2);
                    }
                }
            }

            return result;
        }
    }
}