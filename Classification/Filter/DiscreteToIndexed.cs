using Classification.Attribute;

namespace Classification.Filter
{
    public class DiscreteToIndexed : LaryFilter
    {
        /**
         * <summary> Constructor for discrete to indexed filter.</summary>
         *
         * <param name="dataSet">The dataSet whose instances whose discrete attributes will be converted to indexed attributes</param>
         */
        public DiscreteToIndexed(DataSet.DataSet dataSet) : base(dataSet)
        {
        }

        /**
         * <summary> Converts discrete attributes of a single instance to indexed version.</summary>
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
                    instance.AddAttribute(new DiscreteIndexedAttribute(instance.GetAttribute(i).ToString(), index,
                        attributeDistributions[i].Count));
                }
            }

            RemoveDiscreteAttributes(instance, size);
        }

        /**
         * <summary> Converts the data definition with discrete attributes, to data definition with DISCRETE_INDEXED attributes.</summary>
         */
        protected override void ConvertDataDefinition()
        {
            var dataDefinition = dataSet.GetDataDefinition();
            var size = dataDefinition.AttributeCount();
            for (var i = 0; i < size; i++)
            {
                if (attributeDistributions[i].Count > 0)
                {
                    dataDefinition.AddAttribute(AttributeType.DISCRETE_INDEXED);
                }
            }

            RemoveDiscreteAttributes(size);
        }
    }
}