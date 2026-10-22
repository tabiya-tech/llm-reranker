whether this **specific job advertisement** requires
  education **beyond Kenya secondary school** (KCSE / Form 4 / O-Level completion).

  Set **`true`** when the listing explicitly requires any of:
  - Certificate, diploma, higher diploma, or degree from a college, university, or TVET institution
  - Named professional qualifications (e.g. CPA, ACCA, nursing registration, teaching certificate
    from a post-secondary institution)
  - "Graduate", "bachelor's", "master's", or equivalent post-secondary credentials

  Set **`false`** when:
  - Only KCSE, O-Level, Form 4, or "secondary education" is required or mentioned
  - Only a KCSE grade is specified (e.g. "KCSE C+", "D plain and above") with no college/university qualification
  - No education requirement is stated, or requirements are experience/skills only
  - Requirements are unclear or ambiguous — prefer **`false`** rather than guessing

  Do **not** treat generic phrases like "minimum qualification" or "relevant qualification" as post-secondary
  unless the ad clearly points to certificate/diploma/degree-level study after secondary school.
