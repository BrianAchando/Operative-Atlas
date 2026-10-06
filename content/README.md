# Editing COVA's text

Every operation has one file in `content/procedures/<group>/<operation>.md`. You can edit it on GitHub in the
browser (open the file, click the pencil, then "Propose changes"). When the change is merged, COVA rebuilds
and goes live by itself in a few minutes.

## What a file looks like

```
---
id: vsd-pm                      ← do not change
operation: Ventricular septal defect closure
approach: Perimembranous VSD: transatrial patch closure
summary: One line shown under the menu.
---

## [vsd-patho] Pathophysiology: ventricular septal defect      ← keep the [step-id]; edit the title freely

Ordinary text. **Bold**, *italic*, lists:

- one
- two

| Table | like this |
| --- | --- |
| a | b |

Chain: Large left-to-right shunt → High pulmonary flow → **Eisenmenger**     ← a cause-and-effect chain; bold = the key link

> **Evidence:** the guideline, trial or series that supports the step, with its numbers.

### Case

The vignette shown above the question.

### Question

**Q:** The question?
- [x] The right answer
  > Why it is right (shown after answering).
- [ ] A wrong answer
- [ ] Another wrong answer

## Sources

- [Author, title, journal year](https://link)
```

## Rules the build checks (it refuses to publish if any fail)

- Each `## [step-id]` must match a step of that operation in the 3D atlas. Do not rename or delete them.
- Each question has at least two answers and exactly one marked `[x]`.
- Answers are shuffled for the reader, so their order here does not matter.

## Evidence standard

Every clinical statement needs a source a reader can check: a guideline, trial or series, cited in the
`> **Evidence:**` line and listed under `## Sources`. Unit practice is marked as **KNH practice**.

## What lives elsewhere

Views, structures shown, actions and 3D models come from the pipeline (`pipeline/*.py`) and are stored in
`content/_base/procedures.base.json`. Do not edit that file by hand. New operations and steps are added to the
pipeline first; `node scripts/content-export.mjs` then adds their text here without touching existing text.
