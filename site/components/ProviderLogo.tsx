import type { ComponentType } from "react";
// Direct subpath imports so only the icons we actually use ship to the
// client, instead of the whole @lobehub/icons barrel.
import Ai2 from "@lobehub/icons/es/Ai2/components/Color";
import Ai21 from "@lobehub/icons/es/Ai21/components/Mono";
import Anthropic from "@lobehub/icons/es/Anthropic/components/Mono";
import Aws from "@lobehub/icons/es/Aws/components/Color";
import Baidu from "@lobehub/icons/es/Baidu/components/Color";
import ByteDance from "@lobehub/icons/es/ByteDance/components/Color";
import Cohere from "@lobehub/icons/es/Cohere/components/Color";
import DeepSeek from "@lobehub/icons/es/DeepSeek/components/Color";
import Google from "@lobehub/icons/es/Google/components/Color";
import HuggingFace from "@lobehub/icons/es/HuggingFace/components/Color";
import Hunyuan from "@lobehub/icons/es/Hunyuan/components/Color";
import IBM from "@lobehub/icons/es/IBM/components/Mono";
import InternLM from "@lobehub/icons/es/InternLM/components/Color";
import Meta from "@lobehub/icons/es/Meta/components/Color";
import Microsoft from "@lobehub/icons/es/Microsoft/components/Color";
import Minimax from "@lobehub/icons/es/Minimax/components/Color";
import Mistral from "@lobehub/icons/es/Mistral/components/Color";
import Moonshot from "@lobehub/icons/es/Moonshot/components/Mono";
import NousResearch from "@lobehub/icons/es/NousResearch/components/Mono";
import Nvidia from "@lobehub/icons/es/Nvidia/components/Color";
import OpenAI from "@lobehub/icons/es/OpenAI/components/Mono";
import OpenChat from "@lobehub/icons/es/OpenChat/components/Color";
import Qwen from "@lobehub/icons/es/Qwen/components/Color";
import Rwkv from "@lobehub/icons/es/Rwkv/components/Color";
import Snowflake from "@lobehub/icons/es/Snowflake/components/Color";
import Stability from "@lobehub/icons/es/Stability/components/Color";
import Stepfun from "@lobehub/icons/es/Stepfun/components/Color";
import Together from "@lobehub/icons/es/Together/components/Color";
import Upstage from "@lobehub/icons/es/Upstage/components/Color";
import XAI from "@lobehub/icons/es/XAI/components/Mono";
import XiaomiMiMo from "@lobehub/icons/es/XiaomiMiMo/components/Mono";
import Yi from "@lobehub/icons/es/Yi/components/Color";
import ZAI from "@lobehub/icons/es/ZAI/components/Mono";
import Zhipu from "@lobehub/icons/es/Zhipu/components/Color";

type IconComponent = ComponentType<{ size?: number | string }>;

/**
 * Provider column values (as they appear in the CSVs) mapped to lobehub icons.
 * Monochrome icons inherit currentColor from the wrapper.
 */
const ICON_MAP: Record<string, IconComponent> = {
  Anthropic: Anthropic,
  OpenAI: OpenAI,
  Google: Google,
  xAI: XAI,
  Meta: Meta,
  Mistral: Mistral,
  DeepSeek: DeepSeek,
  Alibaba: Qwen,
  Moonshot: Moonshot,
  "Zhipu AI": Zhipu,
  "Z.ai": ZAI,
  Tencent: Hunyuan,
  Nvidia: Nvidia,
  Amazon: Aws,
  Microsoft: Microsoft,
  Cohere: Cohere,
  MiniMax: Minimax,
  StepFun: Stepfun,
  Ai2: Ai2,
  IBM: IBM,
  Xiaomi: XiaomiMiMo,
  Baidu: Baidu,
  HuggingFace: HuggingFace,
  "01 AI": Yi,
  "AI21 Labs": Ai21,
  Bytedance: ByteDance,
  NousResearch: NousResearch,
  InternLM: InternLM,
  "Together AI": Together,
  Snowflake: Snowflake,
  "Stability AI": Stability,
  "Upstage AI": Upstage,
  RWKV: Rwkv,
  OpenChat: OpenChat,
};

interface ProviderLogoProps {
  provider: string;
  size?: number;
}

export function ProviderLogo({ provider, size = 18 }: ProviderLogoProps) {
  const Icon = ICON_MAP[provider];

  if (!Icon) {
    // Neutral monogram fallback for unknown providers.
    return (
      <span
        aria-hidden
        className="inline-flex shrink-0 items-center justify-center rounded-md border border-cardborder bg-card font-display font-semibold text-muted"
        style={{ width: size, height: size, fontSize: Math.round(size * 0.55) }}
      >
        {(provider.trim()[0] ?? "?").toUpperCase()}
      </span>
    );
  }

  return (
    <span
      aria-hidden
      className="inline-flex shrink-0 items-center justify-center text-fg"
      style={{ width: size, height: size }}
    >
      <Icon size={size} />
    </span>
  );
}
