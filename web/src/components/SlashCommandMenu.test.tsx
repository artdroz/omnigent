import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { skillMenuDescription, SlashCommandMenu } from "./SlashCommandMenu";

afterEach(cleanup);

const props = {
  query: "",
  activeIndex: -1,
  onSelect: vi.fn(),
  commands: { "/help": "Show help" },
};

describe("skill discovery state", () => {
  it("keeps Commands usable while the Skills section loads", () => {
    const onSelect = vi.fn();
    render(<SlashCommandMenu {...props} skillsStatus="loading" onSelect={onSelect} />);
    expect(screen.getByText("Commands")).toBeVisible();
    expect(screen.getByText("Skills")).toBeVisible();
    expect(screen.getByRole("status")).toHaveTextContent("Loading skills…");
    fireEvent.click(screen.getByTestId("slash-menu-item-help"));
    expect(onSelect).toHaveBeenCalledWith("/help");
  });

  it("shows loading with no matches and fills the open menu when skills arrive", () => {
    const { rerender } = render(
      <SlashCommandMenu {...props} query="review" skillsStatus="loading" />,
    );
    expect(screen.getByRole("status")).toHaveTextContent("Loading skills…");
    rerender(
      <SlashCommandMenu
        {...props}
        query="review"
        skillsStatus="ready"
        commands={{ ...props.commands, "/code-review": "Review code" }}
      />,
    );
    expect(screen.queryByRole("status")).toBeNull();
    expect(screen.getByTestId("slash-menu-item-code-review")).toBeVisible();
  });

  it("stops loading when discovery successfully finds no skills", () => {
    const { rerender } = render(<SlashCommandMenu {...props} skillsStatus="loading" />);
    rerender(<SlashCommandMenu {...props} skillsStatus="ready" />);
    expect(screen.getByRole("status")).toHaveTextContent("No skills available");
    expect(screen.queryByText("Loading skills…")).toBeNull();
  });

  it("keeps cached skills visible during a refresh or failure and offers retry", () => {
    const onRetrySkills = vi.fn();
    const commands = { "/review": "Review code" };
    const { rerender } = render(
      <SlashCommandMenu {...props} commands={commands} skillsStatus="loading" />,
    );
    expect(screen.getByTestId("slash-menu-item-review")).toBeVisible();
    rerender(
      <SlashCommandMenu
        {...props}
        commands={commands}
        skillsStatus="error"
        onRetrySkills={onRetrySkills}
      />,
    );
    expect(screen.getByTestId("slash-menu-item-review")).toBeVisible();
    expect(screen.queryByText("Loading skills…")).toBeNull();
    fireEvent.click(screen.getByRole("button", { name: "Retry" }));
    expect(onRetrySkills).toHaveBeenCalledOnce();
  });

  it("shows disconnected state without a spinner", () => {
    render(<SlashCommandMenu {...props} skillsStatus="unavailable" />);
    expect(screen.getByRole("status")).toHaveTextContent("Skills unavailable while disconnected.");
  });

  it("preserves the existing empty-menu behavior when discovery state is absent", () => {
    const { container } = render(<SlashCommandMenu {...props} commands={{}} />);
    expect(container).toBeEmptyDOMElement();
  });
});

describe("skill display names", () => {
  it("types the directory name and labels the row with the frontmatter name", () => {
    const skill = {
      name: "asd-ste100",
      description: "Write in Simplified Technical English",
      display_name: "Simplified Technical English (ASD-STE100)",
    };
    const onSelect = vi.fn();
    render(
      <SlashCommandMenu
        {...props}
        query="asd"
        onSelect={onSelect}
        commands={{ "/asd-ste100": skillMenuDescription(skill) }}
      />,
    );
    const row = screen.getByTestId("slash-menu-item-asd-ste100");
    expect(row).toHaveTextContent(
      "/asd-ste100Simplified Technical English (ASD-STE100) — Write in Simplified Technical English",
    );
    fireEvent.click(row);
    expect(onSelect).toHaveBeenCalledWith("/asd-ste100");
  });

  it("omits a display name that adds nothing beyond the command", () => {
    expect(skillMenuDescription({ name: "review", description: "Review code" })).toBe(
      "Review code",
    );
    expect(
      skillMenuDescription({ name: "review", description: "Review code", display_name: "review" }),
    ).toBe("Review code");
    expect(skillMenuDescription({ name: "ste", description: "", display_name: "STE" })).toBe("STE");
  });
});
