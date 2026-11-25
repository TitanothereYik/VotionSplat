using UnrealBuildTool;

public class VotionSplatHostEditorTarget : TargetRules
{
	public VotionSplatHostEditorTarget(TargetInfo Target) : base(Target)
	{
		Type = TargetType.Editor;
		DefaultBuildSettings = BuildSettingsVersion.V5;
		IncludeOrderVersion = EngineIncludeOrderVersion.Unreal5_5;
		ExtraModuleNames.Add("VotionSplatHost");
	}
}
