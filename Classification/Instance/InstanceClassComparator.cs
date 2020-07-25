using System;
using System.Collections.Generic;

namespace Classification.Instance
{
    public class InstanceClassComparator : IComparer<Instance>
    {
        /**
         * <summary> Compares two {@link Instance} inputs and returns a positive value if the first input's class label is greater
         * than the second's class label input lexicographically.</summary>
         *
         * <param name="o1">First {@link Instance} to be compared.</param>
         * <param name="o2">Second {@link Instance} to be compared.</param>
         * <returns>Negative value if the class label of the first instance is less than the class label of the second instance.
         * Positive value if the class label of the first instance is greater than the class label of the second instance.
         * 0 if the class label of the first instance is equal to the class label of the second instance.</returns>
         */
        public int Compare(Instance x, Instance y)
        {
            return string.Compare(x.GetClassLabel(), y.GetClassLabel(), StringComparison.Ordinal);
        }
    }
}