namespace TextAnalysis.Translation;

using System.Buffers;
using IsoEnums.Iso639;
using Microsoft.Extensions.Logging;
using Neco.Common;
using Neco.Common.Extensions;

[Flags]
public enum XXX {
	Unknown = 0,
	UnknownSourceLanguage = 1,
	UnknownTargetLanguage = 2,

	// 8
	// 16
	// 32
	Available = 64,
}

public interface ITranslator : IDisposable {
	public Language[] From { get; }

	public Language[] To { get; }
	public XXX CanTranslate(Language from, Language to);

	public Boolean CanBeUsedConcurrently { get; }

	public String Translate(ReadOnlySpan<Char> text, Language from, Language to);
	public String Translate(ReadOnlySpan<Byte> utf8Bytes, Language from, Language to);

	public IEnumerable<String> Translate(String[] text, Language from, Language to);
	public IEnumerable<String> Translate(ReadOnlySpan<Byte> utf8Bytes, ReadOnlySpan<Int32> splits, Language from, Language to);

	public SelfTestResults SelfTest();
}

public sealed class MultiTranslatorConfiguration {
	public String? FolderScan { get; init; }

	public IEnumerable<ITranslator> Translators { get; }

	public MultiTranslatorConfiguration(IEnumerable<ITranslator> translators) {
		Translators = translators;
	}

	public MultiTranslatorConfiguration(String folderScan, IEnumerable<ITranslator> translators) {
		FolderScan = folderScan;
		Translators = translators;
	}
}

public sealed class MultiTranslator : ITranslator {
	private readonly ILogger<MultiTranslator> _logger;
	private readonly Dictionary<Language, List<ITranslator>> _translators = new();

	/// <inheritdoc />
	public Language[] From => _translators.Keys.ToArray();

	/// <inheritdoc />
	public Language[] To => _translators.SelectMany(kv => kv.Value).SelectMany(t => t.To).Distinct().ToArray();

	/// <inheritdoc />
	public Boolean CanBeUsedConcurrently => true;

	public MultiTranslator(MultiTranslatorConfiguration config, ILogger<MultiTranslator> logger) {
		ArgumentNullException.ThrowIfNull(config);
		_logger = logger;
		if (config.FolderScan != null)
			FolderScan(config.FolderScan);

		foreach (ITranslator translator in config.Translators) {
			AddTranslator(translator);
		}
	}

	public void AddTranslator(ITranslator translator) {
		foreach (Language language in translator.From) {
			_logger.LogInformation("Using pre-instanced translator {LanguagesFrom} -> {LanguagesTo}: {Translator}", String.Join(", ", translator.From.Select(l => $"{l}({l.Get3Code()})")), String.Join(", ", translator.To.Select(l => $"{l}({l.Get3Code()})")), translator);
			if (_translators.TryGetValue(language, out List<ITranslator>? translators)) {
				translators.Add(translator);
			} else {
				_translators.Add(language, [translator]);
			}
		}
	}

	public void FolderScan(String rootFolder) {
		_logger.LogInformation("Folder scan for translation models: {Directory} in {DirectoryAbsloute}", rootFolder, Path.GetFullPath(rootFolder));
	}

	private ITranslator GetTranslatorOrThrow(Language from, Language to) {
		if (!_translators.TryGetValue(from, out List<ITranslator>? translators)) {
			throw new ArgumentNullException(nameof(from), $"Unable to translate from {from} to {to}. No translators available for source language {from}.");
		}

		ITranslator? translator = translators.FirstOrDefault(t => t.CanTranslate(from, to) == XXX.Available);
		if (translator == null) {
			throw new ArgumentNullException(nameof(to), $"Unable to translate from {from} to {to}. No translators available for target language {to}.");
		}

		return translator;
	}

	/// <inheritdoc />
	public XXX CanTranslate(Language from, Language to) {
		if (_translators.Any(t => t.Value.Any(t => t.CanTranslate(from, to) == XXX.Available)))
			return XXX.Available;

		return XXX.Unknown;
	}

	/// <inheritdoc />
	public String Translate(ReadOnlySpan<Char> text, Language from, Language to) {
		unsafe {
			Byte[] utf8 = new byte[MagicNumbers.Utf8NoBom.GetMaxByteCount(text.Length)];
			fixed (byte* ptrBytes = utf8)
			fixed (char* ptr = text) {
				Int32 bytesWritten = MagicNumbers.Utf8NoBom.GetBytes(ptr, text.Length, ptrBytes, utf8.Length);
				return Translate(utf8.AsSpan(0, bytesWritten), from, to);
			}
		}
	}

	/// <inheritdoc />
	public String Translate(ReadOnlySpan<Byte> utf8Bytes, Language from, Language to) {
		ITranslator translator = GetTranslatorOrThrow(from, to);

		return translator.Translate(utf8Bytes, from, to);
	}

	/// <inheritdoc />
	public IEnumerable<String> Translate(String[] text, Language from, Language to) {
		return text.Where(txt =>!String.IsNullOrWhiteSpace(txt)).Select(txt => Translate(txt, from, to));
	}

	/// <inheritdoc />
	public IEnumerable<String> Translate(ReadOnlySpan<Byte> utf8Bytes, ReadOnlySpan<Int32> splits, Language from, Language to) {
		ITranslator translator = GetTranslatorOrThrow(from, to);
		
		return translator.Translate(utf8Bytes, splits, from, to);
		
		// List<String> result = new List<String>(splits.Length);
		// Int32 offset = 0;
		// for (int i = 0; i < splits.Length; i++) {
		// 	Int32 split = splits[i];
		// 	result.Add(Translate(utf8Bytes.Slice(offset, split - offset), from, to));
		// 	offset = split;
		// }
		//
		// return result;
	}

	/// <inheritdoc />
	public SelfTestResults SelfTest() {
		SelfTestResults results = new(true, String.Empty, String.Empty);

		foreach (ITranslator translator in _translators.SelectMany(kv => kv.Value).Distinct()) {
			SelfTestResults result = translator.SelfTest();
			results.CombineWith(result, $"{translator.GetType().GetName()} [{String.Join(", ", translator.From.Select(l => l.Get3Code()))}]->[{String.Join(", ", translator.To.Select(l => l.Get3Code()))}]");
		}

		return results;
	}

	#region IDisposable

	/// <inheritdoc />
	public void Dispose() {
	}

	#endregion
}

// public class SingleTranslator : ITranslator {}