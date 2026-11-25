using UnrealBuildTool;

public class VotionSplatHostTarget : TargetRules
{
	public VotionSplatHostTarget(TargetInfo Target) : base(Target)
	{
		Type = TargetType.Game;
		DefaultBuildSettings = BuildSettingsVersion.V5;
		IncludeOrderVersion = EngineIncludeOrderVersion.Unreal5_5;
		ExtraModuleNames.Add("VotionSplatHost");
	}
}
