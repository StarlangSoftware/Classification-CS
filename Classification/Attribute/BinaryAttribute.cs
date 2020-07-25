namespace Classification.Attribute
{
    public class BinaryAttribute : DiscreteAttribute
    {
        /**
         * <summary>Constructor for a binary discrete attribute. The attribute can take only two values "True" or "False".</summary>
         *
         * <param name="value">Value of the attribute. Can be true or false.</param>
         */
        public BinaryAttribute(bool value) : base(value.ToString())
        {
        }
    }
}