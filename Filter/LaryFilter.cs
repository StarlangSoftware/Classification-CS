using System.Collections.Generic;
using Math;

namespace Classification.Filter
{
    public abstract class LaryFilter : FeatureFilter
    {
        protected List<DiscreteDistribution> attributeDistributions;

        /**
         * <summary> Constructor that sets the dataSet and all the attributes distributions.</summary>
         *
         * <param name="dataSet">DataSet that will be used.</param>
         */
        public LaryFilter(DataSet.DataSet dataSet) : base(dataSet) {
            attributeDistributions = dataSet.GetInstanceList().AllAttributesDistribution();
        }

        /**
         * <summary> The removeDiscreteAttributes method takes an {@link Instance} as an input, and removes the discrete attributes from
         * given instance.</summary>
         *
         * <param name="instance">Instance to removes attributes from.</param>
         * <param name="size">    Size of the given instance.</param>
         */
        protected void RemoveDiscreteAttributes(Instance.Instance instance, int size) {
            var k = 0;
            for (var i = 0; i < size; i++) {
                if (attributeDistributions[i].Count > 0) {
                    instance.RemoveAttribute(k);
                } else {
                    k++;
                }
            }
        }

        /**
         * <summary> The removeDiscreteAttributes method removes the discrete attributes from dataDefinition.</summary>
         *
         * <param name="size">Size of item that attributes will be removed.</param>
         */
        protected void RemoveDiscreteAttributes(int size) {
            var dataDefinition = dataSet.GetDataDefinition();
            var k = 0;
            for (var i = 0; i < size; i++) {
                if (attributeDistributions[i].Count > 0) {
                    dataDefinition.RemoveAttribute(k);
                } else {
                    k++;
                }
            }
        }
        
    }
}