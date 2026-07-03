# BISK_RFv4 Project Architecture

## Philosophy

BISK_RFv4 is designed as a modular Django ERP.

Each application owns its own domain.

Applications communicate through well-defined relationships.

The project should remain maintainable for many years.

---

# Project Structure

```
apps/
    identity/
    attendance/
    academics/
    finance/
    transport/
    hr/
    library/
    ...
```

Each app should have:

- models.py
- admin.py
- migrations/
- services.py (when business logic grows)
- selectors.py (for complex queries)
- validators.py (when needed)

---

# Ownership

Each model belongs to one app.

Avoid circular ownership.

Example:

Identity owns Person.

Attendance references Person.

Attendance never owns Person.

---

# Business Logic

Business rules should not live inside views.

Prefer:

View
↓

Service

↓

Models

---

# Database

Every migration should represent one logical feature.

Avoid unrelated schema changes in the same migration.

---

# Future

The project is expected to expand.

Architecture should favor extension over modification.

Adding a new module should require minimal changes to existing modules.
