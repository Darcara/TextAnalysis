namespace TextAnalysis.Translation;

using System.Diagnostics.CodeAnalysis;
using System.Text;

public sealed class SelfTestResults {
	[MemberNotNullWhen(true, nameof(Input), nameof(Output))]
	public Boolean Success { get; private set; }

	public String? Input { get; private set; }
	public String? Output { get; private set; }
	public StringBuilder Log { get; init; }

	public SelfTestResults(Boolean success, String? input, String? output, StringBuilder? log = null) {
		if (success) {
			ArgumentNullException.ThrowIfNull(input, nameof(input));
			ArgumentNullException.ThrowIfNull(output, nameof(output));
		}
		
		Success = success;
		Input = input;
		Output = output;
		Log = log ?? new StringBuilder();
	}

	public static SelfTestResults Fail(String message) => new(false, null, null, new StringBuilder(message));

	public void CombineWith(SelfTestResults result, String? name) {
		Success = Success && result.Success;
		Boolean isFirst = Log.Length == 0;
		String header = $"{(isFirst ? String.Empty : Environment.NewLine)}" +
		                $"*******************************************{Environment.NewLine}" +
		                (name ?? "A new test begins...") + Environment.NewLine + 
		                $"Testresult: {(result.Success ? "Success" : "Failure")} - Overall: {(Success ? "Success" : "Failure")}{Environment.NewLine}" +
		                $"*******************************************{Environment.NewLine}";
		Log.Append(header);
		Log.Append(result.Log);
		
		Input += header + result.Input;
		Output += header + result.Output;
	}

	#region Overrides of Object

	/// <inheritdoc />
	public override String ToString() => Log.ToString();

	#endregion
}