from .agent import EmailPlan, EmailSkeleton, EmailStep, _build_executor_prompt


PRESET_PLANS = {
    "black-friday-fashion": [
        ("VAULTURA Black Friday: Early Access", [
            ("Your invitation arrives", "Something extraordinary arrives Friday"),
            ("The VAULT30 reveal", "Black Friday is here: take 30% off"),
            ("The final edit", "24 hours left to shop VAULT30"),
        ]),
        ("VAULTURA Black Friday: The Curated Edit", [
            ("A first look", "Your next wardrobe chapter starts Friday"),
            ("The collection is open", "The Black Friday edit: 30% off with VAULT30"),
            ("Your last chance", "The pieces you love won't wait"),
        ]),
        ("VAULTURA Black Friday: The Countdown", [
            ("48 hours to go", "Consider this your exclusive heads-up"),
            ("The wait is over", "VAULT30 is live. Your Black Friday starts now."),
            ("The clock is ticking", "One final day. One exceptional offer."),
        ]),
    ],
    "saas-trial-nurture": [
        ("Taskflow Trial: Your First Wins", [
            ("Your trial starts here", "Your Taskflow trial is live. Let's get started."),
            ("Build your daily rhythm", "Three small steps for a more productive day"),
            ("Meet Automation", "Give your team five hours back each week"),
            ("Keep your momentum", "Your trial is ending: keep building with 20% off"),
        ]),
        ("Taskflow Trial: From Setup to Success", [
            ("Set up your first project", "Your first Taskflow win is five minutes away"),
            ("Make Taskflow your own", "Get more from your trial with these three tips"),
            ("Let busywork run itself", "Meet your team's next productivity upgrade"),
            ("Make the next move", "Keep your team moving: save 20% for three months"),
        ]),
        ("Taskflow Trial: Work Better Together", [
            ("Welcome to your workspace", "Your team has a new home in Taskflow"),
            ("Find your team's flow", "A few ways to make this week run smoother"),
            ("Unlock your team's potential", "Less admin. More progress. Meet Automation."),
            ("Keep the team connected", "Your trial ends soon. Your next chapter starts here."),
        ]),
    ],
    "streaming-onboarding": [
        ("StreamVault: Welcome Aboard", [
            ("Your account is ready", "Welcome to StreamVault. Your next watch awaits."),
            ("Reset your password", "Reset your StreamVault password"),
            ("Premium is active", "You're all set: StreamVault Premium is active"),
        ]),
        ("StreamVault: Your First Watch", [
            ("Find your next favourite", "Your StreamVault account is ready to explore"),
            ("Secure password reset", "Your StreamVault password reset request"),
            ("Everything in your plan", "Welcome to Premium. Here's what's included."),
        ]),
        ("StreamVault: Ready, Set, Stream", [
            ("Start something great", "Tonight's entertainment starts with StreamVault"),
            ("Get back into your account", "Change your StreamVault password securely"),
            ("Your subscription, confirmed", "Your StreamVault Premium subscription is confirmed"),
        ]),
    ],
}


def build_preset_plan(
    preset_id: str,
    goal: str,
    sections: list[list[str]],
    variation: int,
) -> EmailPlan:
    title, steps = PRESET_PLANS[preset_id][variation]
    if len(steps) != len(sections):
        raise ValueError("Preset plan and section counts do not match.")
    return EmailPlan(sequence_title=title, preset_id=preset_id, emails=[
        EmailStep(
            step=index + 1,
            title=step_title,
            subject_line=subject,
            body_section_count=len(sections[index]),
            agent_prompt=_build_executor_prompt(
                EmailSkeleton(step=index + 1, title=step_title, subject_line=subject),
                title, goal, total_emails=len(steps), sections=sections[index],
            ),
        )
        for index, (step_title, subject) in enumerate(steps)
    ])