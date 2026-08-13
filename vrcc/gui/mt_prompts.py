"""Combo greying for the active translation model.

The MT counterpart to :mod:`vrcc.gui.model_prompts`, which does the same job
for the voice model and is scoped to it. A family that renders two languages
with one control token cannot tell them apart, so offering both as targets
would promise a translation it silently cannot deliver: m2m100 and madlad
return Simplified for a Chinese Traditional request.

The decision helper is Qt-free; only the greying touches a widget, and it
imports nothing from PySide6 at module level so this stays importable headless.
No deadlock to guard against, unlike the voice-model case: the model combo in
Settings is never greyed, so the way out is always reachable.
"""

from __future__ import annotations

from vrcc.gui.model_labels import mt_display_name
from vrcc.i18n import tr, tr_noop
from vrcc.translate.registry import MT_MODELS, collapses_onto

_TARGET_COLLAPSED_TIP = tr_noop(
    "{name} writes {language} and {other} the same way. "
    "Choose another translation model first."
)


def collapsed_target(model_id: str, display: str) -> str | None:
    """The language ``display`` would be indistinguishable from under the model
    ``model_id``, or ``None`` when the model renders it distinctly.

    An unknown model id (hand-edited config) restricts nothing, matching the
    escape hatch on the voice-model side.
    """
    spec = MT_MODELS.get(model_id)
    if spec is None:
        return None
    return collapses_onto(spec.family, display)


def target_pairs(
    model_id: str, targets: list[str], source: str
) -> list[tuple[str, str]]:
    """``(asked, shown)`` per surviving target, in order.

    The same rule :func:`vrcc.translate.registry.distinct_targets` applies, so
    the pills promise exactly what the engine will send: a target is resolved
    onto the language the model can actually write, then dropped if that
    duplicates an earlier one or equals the source (translating a language into
    itself just re-sends the original).

    Paired rather than returned as two lists because resolving can drop
    entries, and a caller indexing a shortened display list against the stored
    one shows slot i the name of a different target and then writes that name
    back as the user's choice.

    Kept out of :func:`usable_targets`, whose contract is that it never empties
    a non-empty list: the source rule legitimately can, and the wizard's combo
    greying depends on the softer promise.
    """
    pairs: list[tuple[str, str]] = []
    seen: set[str] = set()
    for asked in targets:
        display = collapsed_target(model_id, asked) or asked
        if display in seen or display == source:
            continue
        seen.add(display)
        pairs.append((asked, display))
    return pairs


def usable_targets(model_id: str, targets: list[str]) -> list[str]:
    """``targets`` with each entry the active model cannot render distinctly
    replaced by the one it collapses onto, then deduplicated in first-seen order.

    Substituted rather than dropped, because dropping empties the list for a
    user whose ONLY target is the collapsed one: a Taiwanese user on the default
    model asks for Chinese Traditional and gets no translation at all, with the
    greying preventing them from choosing it back. Rewriting the label to the
    language the model actually produces keeps translation working and keeps the
    caption honest about which script it is. Where both are already listed this
    still just removes the duplicate, which is the case that frees a slot.
    """
    out: list[str] = []
    for target in targets:
        canonical = collapsed_target(model_id, target) or target
        if canonical not in out:
            out.append(canonical)
    return out


def grey_collapsed_targets(combo, model_id: str) -> None:
    """Disable each target entry the active MT model writes identically to
    another, tooltip naming both languages and the way out."""
    item_model = combo.model()
    for i in range(combo.count()):
        item = item_model.item(i)
        if item is None:
            continue
        other = collapsed_target(model_id, combo.itemText(i))
        item.setEnabled(other is None)
        item.setToolTip(
            ""
            if other is None
            else tr(
                _TARGET_COLLAPSED_TIP,
                name=mt_display_name(model_id),
                language=combo.itemText(i),
                other=other,
            )
        )
