using System;
using System.Runtime.Serialization;

namespace EvoLearning
{
    public class Pre
    {
        [System.Diagnostics.Conditional(CompilationConstant.DebugSymbol)]
        public static void Assert(bool condition, params object[] parts)
        {
            if (condition) return;
            var text = parts.Length == 0 ? "" : ConstructMessage(parts);
            if (text.Length == 0)
            {
                text = "Assertion failed.";
            }
            throw new FailedAssertion(text);
        }

        private static string ConstructMessage(object[] parts) {
            var message = "";
            foreach (var part in parts) {
                message += part.ToString() + ", ";
            }
            return message.Substring(0, message.Length - 2);
        }

        [Serializable]
        public class FailedAssertion : Exception
        {
            public FailedAssertion(string message)
                : base(message)
            { }

            protected FailedAssertion(SerializationInfo info, StreamingContext ctxt)
                : base(info, ctxt)
            { }
        }
    }
}