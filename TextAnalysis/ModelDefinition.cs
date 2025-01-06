namespace TextAnalysis;

public record ModelDefinition {
	/// <summary>
	/// ISO-639 two letter language code (Set 1) designating the source language of this model.
	/// </summary>
	/// <example>en</example>
	public required String SourceLanguage { get; init; }

	/// <summary>
	/// ISO-639 two letter language code (Set 1) designating the target language of this model.
	/// </summary>
	/// <example>de</example>
	public required String TargetLanguage { get; init; }

	public String SourceTokenizer { get; init; } = "source.spm";
	public String VocabularyTokenToIdMap { get; init; } = "vocab.json";
	
	public required Int32 SourceMaxTokens { get; init; }

	public String? ModelDirectoryOverride { get; init; }
	public String ModelDirectory => ModelDirectoryOverride ?? $"{SourceLanguage}-{TargetLanguage}";

	public String? EncoderModelNameOverride { get; init; }
	public String EncoderModelName => EncoderModelNameOverride ?? "encoder_model.onnx";
	public String? DecoderModelNameOverride { get; init; }
	public String DecoderModelName => DecoderModelNameOverride ?? "decoder_model.onnx";
	public String? DecoderWithPasthModelNameOverride { get; init; }
	public String DecoderWithPasthModelName => DecoderWithPasthModelNameOverride ?? "decoder_with_past_model.onnx";

	/// <summary>
	/// The input sentence to test that the model and its weights have loaded correctly
	/// </summary>
	public String? SelfTestInput { get; init; }

	/// <summary>
	/// The output sentence to test that the model and its weights have loaded correctly
	/// </summary>
	public String? SelfTestOutput { get; init; }
}