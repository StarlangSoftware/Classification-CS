using Classification.Attribute;

namespace Classification.Filter
{
    public class LaryToBinary : LaryFilter
    {
        /**
         * <summary> Constructor for L-ary discrete to binary discrete filter.</summary>
         *
         * <param name="dataSet">The instances whose L-ary discrete attributes will be converted to binary discrete attributes.</param>
         */
        public LaryToBinary(DataSet.DataSet dataSet) : base(dataSet)
        {
        }

        /**
         * <summary> Converts discrete attributes of a single instance to binary discrete version using 1-of-L encoding. For example, if
         * an attribute has values red, green, blue; this attribute will be converted to 3 binary attributes where
         * red will have the value true false false, green will have the value false true false, and blue will have the value false false true.</summary>
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
                            instance.AddAttribute(new BinaryAttribute(false));
                        }
                        else
                        {
                            instance.AddAttribute(new BinaryAttribute(true));
                        }
                    }
                }
            }

            RemoveDiscreteAttributes(instance, size);
        }

        /**
         * <summary> Converts the data definition with L-ary discrete attributes, to data definition with binary discrete attributes.</summary>
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
                        dataDefinition.AddAttribute(AttributeType.BINARY);
                    }
                }
            }

            RemoveDiscreteAttributes(size);
        }
    }
}