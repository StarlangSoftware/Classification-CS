using Classification.Attribute;

namespace Classification.Filter
{
    public class DiscreteToContinuous : LaryFilter
    {
        /**
         * <summary> Constructor for discrete to continuous filter.</summary>
         *
         * <param name="dataSet">The dataSet whose instances whose discrete attributes will be converted to continuous attributes using
         *                1-of-L encoding.</param>
         */
        public DiscreteToContinuous(DataSet.DataSet dataSet) : base(dataSet)
        {
        }

        /**
         * <summary> Converts discrete attributes of a single instance to continuous version using 1-of-L encoding. For example, if
         * an attribute has values red, green, blue; this attribute will be converted to 3 continuous attributes where
         * red will have the value 100, green will have the value 010, and blue will have the value 001.</summary>
         *
         * <param name="instance">The instance to be converted.</param>
         */
        protected override void ConvertInstance(Instance.Instance instance)
        {
            var size = instance.AttributeSize();
            for (var i = 0; i < size; i++)
            {
                if (attributeDistributions[i].Count > 0)
                {
                    var index = attributeDistributions[i].GetIndex(instance.GetAttribute(i).ToString());
                    for (var j = 0; j < attributeDistributions[i].Count; j++)
                    {
                        if (j != index)
                        {
                            instance.AddAttribute(new ContinuousAttribute(0));
                        }
                        else
                        {
                            instance.AddAttribute(new ContinuousAttribute(1));
                        }
                    }
                }
            }

            RemoveDiscreteAttributes(instance, size);
        }

        /**
         * <summary> Converts the data definition with discrete attributes, to data definition with continuous attributes. Basically,
         * for each discrete attribute with L possible values, L more continuous attributes will be added.</summary>
         */
        protected override void ConvertDataDefinition()
        {
            var dataDefinition = dataSet.GetDataDefinition();
            var size = dataDefinition.AttributeCount();
            for (var i = 0; i < size; i++)
            {
                if (attributeDistributions[i].Count > 0)
                {
                    for (var j = 0; j < attributeDistributions[i].Count; j++)
                    {
                        dataDefinition.AddAttribute(AttributeType.CONTINUOUS);
                    }
                }
            }

            RemoveDiscreteAttributes(size);
        }
    }
}