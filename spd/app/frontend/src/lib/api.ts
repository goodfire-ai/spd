const API_URL = "http://localhost:8000";

export type OutputTokenLogit = {
  token: string;
  logit: number;
  probability: number;
};

export type TokenCIs = {
  l0: number;
  component_cis: number[];
  indices: number[];
};

export type LayerCIs = {
  module: string;
  token_cis: TokenCIs[];
};

export type RunPromptResponse = {
  prompt_tokens: string[];
  layer_cis: LayerCIs[];
  full_run_token_logits: OutputTokenLogit[][];
  ci_masked_token_logits: OutputTokenLogit[][];
};

export type ComponentMask = Record<string, number[][]>;

export type ModifyComponentsResponse = {
  token_logits: OutputTokenLogit[][];
};

class ApiClient {
  constructor(private apiUrl: string = API_URL) {}

  async runPrompt(prompt: string): Promise<RunPromptResponse> {
    const response = await fetch(`${this.apiUrl}/run`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify({ prompt })
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || "Failed to run prompt");
    }

    const data = await response.json();
    data.layer_cis.reverse();
    return data;
  }

  async modifyComponents(
    prompt: string,
    componentMask: ComponentMask
  ): Promise<ModifyComponentsResponse> {
    const response = await fetch(`${this.apiUrl}/modify`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify({
        prompt,
        component_mask: componentMask
      })
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || "Failed to modify components");
    }

    return response.json();
  }

  async loadRun(wandbRunId: string): Promise<void> {
    const response = await fetch(`${this.apiUrl}/load`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify({ wandb_run_id: wandbRunId })
    });
    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || "Failed to load run");
    }
  }
}

export const api = new ApiClient();
