namespace TextAnalysis.Translation;

using System.Diagnostics;
using System.Numerics.Tensors;
using System.Text;
using IsoEnums.Iso639;
using Microsoft.ML.OnnxRuntime;
using SentencePieceTokenizer;

[DebuggerDisplay("{ModelDefinition.SourceLanguage}-{ModelDefinition.TargetLanguage}")]
internal class TranslationModel2WithPast : AConfigurationBasedTranslator {

	public TranslationModel2WithPast(ModelDefinition modelDefinition, MarianTokenizer tokenizer, OnnxSession encoderSession, OnnxSession decoderWithPastSession) : base(modelDefinition, tokenizer, encoderSession, decoderWithPastSession) { 
	}

	/// <inheritdoc />
	public override SelfTestResults SelfTest() {
		if (String.IsNullOrEmpty(ModelDefinition.SelfTestInput)) return SelfTestResults.Fail("No Input");
		if (String.IsNullOrEmpty(ModelDefinition.SelfTestOutput)) return SelfTestResults.Fail("No Output");

		StringBuilder log = new();
		log.AppendLine($"Testing 2-part {ModelDefinition.EncoderModelName} and {ModelDefinition.DecoderModelName} with {EncoderSession.Configuration.OptimizationLevel}");
		(Int64[] encodedIds, Int64[] encodedAttention) = SelfTestLogInput(log);

		using OrtValue? inputOrtValue = OrtValue.CreateTensorValueFromMemory(encodedIds, [1, encodedIds.Length]);
		using OrtValue? encodedAttentionOrtValue = OrtValue.CreateTensorValueFromMemory(encodedAttention, [1, encodedIds.Length]);

		Dictionary<String, OrtValue> inputs = EncoderSession.CreateStaticInputs(3);
		using OrtValue bestTokenInput = OrtValue.CreateTensorValueFromMemory([Tokenizer.PadToken], [1, 1]);
		inputs.Add("input_ids", inputOrtValue);
		inputs.Add("decoder_input_ids", bestTokenInput);
		inputs.Add("attention_mask", encodedAttentionOrtValue);

		using IDisposableReadOnlyCollection<OrtValue>? encoderDecoderOutputs = EncoderSession.Session.Run(EncoderSession.RunOptions, inputs, EncoderSession.Session.OutputNames);

		List<Int64> outputTokens = SelfTestRunDecoderWithPast(encoderDecoderOutputs, bestTokenInput, encodedAttentionOrtValue, encodedIds, encodedAttention, log);
		
		return SelfTestLogResult(log, outputTokens);
	}

	#region Implementation of AConfigurationBasedTranslator

	/// <inheritdoc />
	protected override String InternalTranslate(OrtValue inputOrtValue, OrtValue encodedAttentionOrtValue, Int32 batchSize) {

		Dictionary<String, OrtValue> inputs = EncoderSession.CreateStaticInputs(3);
		using OrtValue bestTokenInput = OrtValue.CreateTensorValueFromMemory([Tokenizer.PadToken], [1, 1]);
		inputs.Add("input_ids", inputOrtValue);
		inputs.Add("decoder_input_ids", bestTokenInput);
		inputs.Add("attention_mask", encodedAttentionOrtValue);

		using IDisposableReadOnlyCollection<OrtValue>? encoderDecoderOutputs = EncoderSession.Session.Run(EncoderSession.RunOptions, inputs, EncoderSession.Session.OutputNames);

		return RunDecoderWithPast(encoderDecoderOutputs, bestTokenInput, encodedAttentionOrtValue, batchSize);
	}

	#endregion
}