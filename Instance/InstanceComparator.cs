using System.Collections.Generic;
using Classification.Attribute;

namespace Classification.Instance
{
    public class InstanceComparator : IComparer<Instance>
    {
        private readonly int _attributeIndex;

        /**
         * <summary> Constructor for instance comparator.</summary>
         *
         * <param name="attributeIndex">Index of the attribute of which two instances will be compared.</param>
         */
        public InstanceComparator(int attributeIndex) {
            this._attributeIndex = attributeIndex;
        }

        /**
         * <summary> Compares two instance on the values of the attribute with index attributeIndex.</summary>
         *
         * <param name="x">First instance to be compared</param>
         * <param name="y">Second instance to be compared</param>
         * <returns>-1 if the attribute value of the first instance is less than the attribute value of the second instance.
         * 1 if the attribute value of the first instance is greater than the attribute value of the second instance.
         * 0 if the attribute value of the first instance is equal to the attribute value of the second instance.</returns>
         */
        public int Compare(Instance x, Instance y)
        {
            if (((ContinuousAttribute) x.GetAttribute(_attributeIndex)).GetValue() < ((ContinuousAttribute) y.GetAttribute(_attributeIndex)).GetValue()) {
                return -1;
            }

            if (((ContinuousAttribute) x.GetAttribute(_attributeIndex)).GetValue() > ((ContinuousAttribute) y.GetAttribute(_attributeIndex)).GetValue()) {
                return 1;
            }

            return 0;
        }
    }
}