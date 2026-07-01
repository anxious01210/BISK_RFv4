# Identity Architecture Options — BISK_RFv4

Date: 2026-06-30
Branch: feature/person-architecture
Mode: read-only architecture design

---

## Objective

Design and compare identity architecture options for introducing Person, CustomUser, PersonRole, StudentProfile, and StaffProfile into BISK_RFv4. The analysis balances short-term migration safety against long-term ERP/LMS architecture quality, considering future portal, mobile, social login, multi-school, and face-recognition-for-all requirements.

---

## Context

### Current State

| Aspect | Status |
|---|---|
| Auth model | `django.contrib.auth.models.User` (default) — **no `AUTH_USER_MODEL` set** |
| Existing login users | **4 users** — staff/supervisors only |
| Student model | Standalone, **no FK/O2O to auth.User** |
| Student→auth.User linkage | **None** — zero relationship |
| FK to auth.User | 3 fields: `WalletTransaction.created_by`, `MealRecord.confirmed_by`, `MealRecord.reversed_by` |
| FK/O2O to Student | 6 models: `AttendanceRecord`, `AttendanceEvent`, `FaceEmbedding`, `MealSubscription`, `Wallet`, `WalletTransaction` |
| Existing migrations | 30+ in `attendance`, all reference `attendance_student` |
| Student records | Many (production data) |
| Wallet/meal records | Many (production data) |
| Auth mechanism | Group-based: `supervisor`, `meal_supervisor`, `api_user` groups on `auth.User` |
| Permissions | No custom permissions. Zero `has_perm()` calls. Zero template-level perms. |

### Long-Term Requirements

- Student portal (students need login accounts)
- Parent portal (parents need accounts linked to students)
- Staff/teacher portal (staff need richer accounts)
- Mobile app API (JWT/OAuth2)
- Microsoft/Google social login
- Wallet/lunch system keyed to Person (not Student)
- Face recognition for students AND staff
- Future multi-school / ERP evolution
- Clean domain model that separates identity from role

### Constraint

Django does not support changing `AUTH_USER_MODEL` after migrations have been run on a database. The setting locks in at first migration that references the user model. Any approach that changes `AUTH_USER_MODEL` on the existing database requires either:
1. A new database and full data migration
2. Manual SQL `ALTER` + `RENAME` operations
3. A custom `db_table` + state migration hack

---

## Options Compared

### Option A: Keep `auth.User` + Person with optional `OneToOneField(User)`

```
auth.User (unchanged, AUTH_USER_MODEL not set)
  ↑
  │ (optional, nullable O2O)
  │
Person
  ├── StudentProfile
  ├── StaffProfile
  └── PersonRole (M2M through)
```

Person holds identity fields (name, h_code, email, phone, person_type). User remains the authentication target. Business logic eventually targets Person. User is only for login.

---

### Option B: Create `CustomUser(AbstractUser)` now, Person links to CustomUser

```
CustomUser(AbstractUser) [AUTH_USER_MODEL = "attendance.CustomUser"]
  ↑
  │ (optional, nullable O2O)
  │
Person
  ├── StudentProfile
  ├── StaffProfile
  └── PersonRole
```

Create a new CustomUser model inheriting from AbstractUser, set it as AUTH_USER_MODEL immediately, then link Person to CustomUser. This requires migrating existing `auth.User` data into `attendance_customuser` and updating all FK references.

---

### Option C: Person IS the custom user model

```
Person(AbstractUser) [AUTH_USER_MODEL = "attendance.Person"]
  ├── StudentProfile
  ├── StaffProfile
  └── PersonRole (or person_type field)
```

Person inherits from AbstractUser directly. One table for identity and authentication. Profiles are thin extensions. This is the canonical "clean architecture" target.

---

### Option D (Recommended Hybrid): Progressive Identity Model

**Phase 1 (now — additive, zero data risk):**

```
auth.User (unchanged)
  ↑
  │ (optional, nullable O2O, synced)
  │
Person (concrete model, all identity + auth-ready fields)
  ├── StudentProfile
  ├── StaffProfile
  ├── ParentProfile (future)
  └── PersonRole (M2M through)
```

Person is a concrete model holding ALL fields it would need to become the user model later (email, password, is_active, is_staff, is_superuser, last_login, date_joined, groups M2M, user_permissions M2M). An optional `auth_user` O2O bridges to the current `auth.User`. A sync signal keeps them in sync. Business logic targets Person from day one.

**Phase 2 (after squash + readiness):**

```
Person(AbstractUser) [AUTH_USER_MODEL = "attendance.Person"]
  ├── StudentProfile
  ├── StaffProfile
  └── PersonRole
```

Remove the `auth_user` bridge FK. Set `AUTH_USER_MODEL`. Copy existing `auth_user` rows into `attendance_person` (all columns already exist from Phase 1). Drop `auth_user` table. Zero schema migration needed for Person — just a Django state migration telling it Person is now AbstractUser.

---

## Detailed Comparison

### 1. Model Structure

| Aspect | Option A | Option B | Option C | Option D |
|---|---|---|---|---|
| Auth model | `auth.User` (default) | `CustomUser(AbstractUser)` | `Person(AbstractUser)` | **Phase 1:** `auth.User` **Phase 2:** `Person(AbstractUser)` |
| Identity model | `Person` (concrete) | `Person` (concrete) | N/A (Person IS the user) | `Person` (concrete → AbstractUser) |
| Person↔User link | `Person.user` O2O (nullable) | `Person.custom_user` O2O (nullable) | N/A (same table) | **P1:** `Person.auth_user` O2O **P2:** dropped |
| Profile models | StudentProfile, StaffProfile O2O to Person | Same | Same | Same |
| Student table | Deprecated over time | Deprecated over time | Deprecated over time | Deprecated over time |
| Unique identifier | `Person.h_code` | `Person.h_code` | `Person.h_code` (or `username`) | `Person.h_code` |

**Key difference:** Option C has ONE identity table. Options A/B have two (User + Person) with a linking FK. Option D has two in Phase 1, one in Phase 2.

---

### 2. Authentication Implications

| Aspect | Option A | Option B | Option C | Option D |
|---|---|---|---|---|
| Login flow | Unchanged (Django auth) | New CustomUser used for login | Person used for login | **P1:** Unchanged **P2:** Person used |
| `request.user` | `auth.User` instance | `CustomUser` instance | `Person` instance | **P1:** `auth.User` **P2:** `Person` |
| Password storage | On `auth.User` | On `CustomUser` | On `Person` | **P1:** On `auth.User` (+ synced to Person) **P2:** On Person |
| Session/auth middleware | Unchanged | Unchanged (User model swap) | Unchanged (User model swap) | Unchanged both phases |
| Social auth (future) | Add backend for `auth.User` | Add backend for `CustomUser` | Add backend for `Person` | **P1:** auth.User **P2:** Person |
| JWT for mobile | Works with any user model | Works | Works | Works both phases |

**Key difference:** Option B and C require the irreversible `AUTH_USER_MODEL` switch NOW. Option A never switches. Option D switches later when safe.

---

### 3. Authorization / Groups / Permissions Implications

| Aspect | Option A | Option B | Option C | Option D |
|---|---|---|---|---|
| Groups table | `auth_group` | `attendance_customuser_groups` (+ migration) | `attendance_person_groups` (+ migration) | **P1:** `auth_group` **P2:** `attendance_person_groups` |
| User→Groups M2M | `auth_user_groups` | `attendance_customuser_groups` | `attendance_person_groups` | **P1:** `auth_user_groups` **P2:** `attendance_person_groups` |
| Permission checks | `user.groups.filter(name=...)` (existing code) | Same pattern, new table | Same pattern, new table | **P1:** unchanged **P2:** new table |
| Custom permissions | Can add on Person or Profile models | Can add on CustomUser | Can add on Person | Can add on Person |
| Role system | `PersonRole` model OR group-based | Same | Same | Same |
| Migration of groups | None needed | Must data-migrate groups to new M2M | Must data-migrate groups to new M2M | **P1:** None **P2:** must migrate |

**Key difference:** Options B/C require migrating group memberships and permissions to new M2M tables NOW. Option A never requires this. Option D defers it to Phase 2.

---

### 4. Migration Difficulty

| Aspect | Option A | Option B | Option C | Option D |
|---|---|---|---|---|
| AUTH_USER_MODEL change | **Not required** (unchanged) | **Required NOW** — highest risk | **Required NOW** — highest risk | **Deferred to Phase 2** |
| Existing migration breakage | **None** (additive) | **Complete** — must squash + fix all 3 `settings.AUTH_USER_MODEL` FK refs | **Complete** — same as B | **None in Phase 1** |
| Data migration for `auth_user` → new table | Not needed | **Required** — copy 4 users to CustomUser | **Required** — copy 4 users to Person | **Deferred to Phase 2** |
| Student → Person+Profile data migration | **Required** (additive) | Required | Required | Required (same in all options) |
| FK migration (6 models: Student→Person) | **Required** (dual-write) | Required | Required | Required |
| Group/permission data migration | Not needed | Required NOW | Required NOW | Deferred to Phase 2 |
| Rollback complexity | **Low** (additive, drop new tables) | **Very high** (schema fundamentally changed) | **Very high** | **Low in P1** |

**Key difference:** Option A has the lowest migration risk by far. Option D defers the high-risk AUTH_USER_MODEL change to when it can be done safely (after squash, on a prepared schema). Options B and C require the high-risk change immediately.

---

### 5. Data Safety

| Aspect | Option A | Option B | Option C | Option D |
|---|---|---|---|---|
| Existing auth.User data | **Preserved** — no change | Must be copied, risk of mismatch | Must be copied, risk of mismatch | **Preserved in P1** — migrated later with preparation |
| Student data | Mirrored to Person+Profile | Mirrored to Person+Profile | Mirrored to Person+Profile | Mirrored to Person+Profile |
| FK integrity during migration | Maintained (dual-write) | Maintained | Maintained | Maintained |
| Rollback | Drop Person/Profile tables | Complex — must restore old schema | Complex — must restore old schema | Drop Person/Profile tables (P1) |
| Production downtime risk | **Low** (additive) | **High** (schema change) | **High** (schema change) | **Low** (P1 additive) |

---

### 6. Portal / Mobile / API Support (Future)

| Aspect | Option A | Option B | Option C | Option D |
|---|---|---|---|---|
| Student login | Must create `auth.User` + link to `Person` | Must create `CustomUser` + link to `Person` | Person IS the user — create Person, done | **P1:** auth.User + Person link **P2:** Person directly |
| `request.user` → Person lookup | `user.person` (if linked) | `user.person` (if linked) | `user` IS the Person | **P1:** `user.person` **P2:** user IS Person |
| API `@login_required` | Works on auth.User | Works on CustomUser | Works on Person | Works both phases |
| API serializer for current user | Must serialize both User + Person | Must serialize both CustomUser + Person | Serialize Person directly | **P1:** both **P2:** Person directly |
| Social auth (Google/MS) | Works with `python-social-auth` + auth.User | Works with CustomUser | Works with Person | Works both phases |
| Parent portal (linked to student) | Person to Person relationship (parent→child) | Same | Same | Same |
| Clean API contract | No (two models exposed) | No (two models exposed) | **Yes** (one identity model) | **P1:** No **P2:** Yes |

**Key difference:** Option C has the cleanest API contract (one identity model = one API representation). Option D achieves this in Phase 2. Options A and B permanently maintain two identity models.

---

### 7. Wallet / Lunch Implications

All options move Wallet, MealSubscription, MealRecord, and WalletTransaction FK from `Student` → `Person`. The key difference is:

| Aspect | Option A | Option B | Option C | Option D |
|---|---|---|---|---|
| Wallet FK target | `Person` | `Person` | `Person` (same as user) | `Person` |
| `Wallet.student` → `Wallet.person` | Required | Required | Required | Required |
| `has_meal` location | `StudentProfile.has_meal` | `StudentProfile.has_meal` | `StudentProfile.has_meal` | `StudentProfile.has_meal` |
| Staff wallet support | Possible (Person has StaffProfile) | Possible | Possible | Possible |

No significant difference between options for wallet/lunch — all must migrate FK from Student to Person.

---

### 8. Face Recognition Implications

| Aspect | Option A | Option B | Option C | Option D |
|---|---|---|---|---|
| FaceEmbedding FK target | `Person` (currently `Student`) | `Person` | `Person` | `Person` |
| Staff face recognition | Person has FaceEmbedding | Same | Same | Same |
| Face matching → identity lookup | `FaceEmbedding.person` → `Person` | Same | Same | Same |
| AttendanceRecord target | `Person` (via StudentProfile or directly) | Same | Same | Same |
| `h_code` lookup | `Person.h_code` (unchanged) | Same | Same | Same |

No significant difference. All options require FaceEmbedding FK to move from Student to Person to support staff face recognition.

---

### 9. Multi-School Readiness

| Aspect | Option A | Option B | Option C | Option D |
|---|---|---|---|---|
| Person-level tenant field | `Person.school` FK (future) | Same | Same | Same |
| Unique h_code across schools | `unique_together = [("school", "h_code")]` | Same | Same | Same |
| User roaming across schools | auth.User is already global | CustomUser is global | Person IS the user — global | **P2:** Person IS the user — global |
| School-specific groups/permissions | Group naming convention or tenant middleware | Same | Same | Same |

No significant difference — all options support `Person.school` FK for multi-school. Option C/D has the advantage that Person identity is inherently cross-school.

---

### 10. Pros

#### Option A: Keep `auth.User`

- **Zero AUTH_USER_MODEL risk** — no irreversible configuration change
- **Additive migration only** — all changes can be rolled back by dropping new tables
- **Existing code untouched** — auth.User-based code continues working
- **No data migration for auth** — 4 existing users stay exactly where they are
- **Group/permission tables untouched** — no M2M data migration
- **Fastest to implement** — can be done incrementally
- **Lowest production risk** — no schema changes to existing tables
- **Easiest rollback** — drop Person/Profile tables, everything reverts

#### Option B: CustomUser now

- None significant — this is the worst of both worlds (two identity models + high migration risk)

#### Option C: Person IS the user model

- **Single identity model** — cleanest long-term architecture
- **`request.user` is Person** — no indirection for portal/mobile code
- **Cleanest API contract** — one model to serialize
- **Simplest portal code** — `user.student_profile` or `user.staff_profile` directly
- **Future-proof** — designed for the target architecture from day one
- **Minimal long-term tech debt** — don't have to maintain a bridge pattern

#### Option D: Progressive Hybrid

- **Lowest Phase 1 risk** (same as Option A) — additive changes only
- **Cleanest Phase 2 outcome** (same as Option C) — single identity model
- **Person is pre-loaded with all auth fields** — Phase 2 requires schema state change, not column additions
- **Business logic targets Person from day one** — code is written for the future model
- **Bridge sync signal keeps auth.User ↔ Person consistent** — no data loss risk
- **Phase 2 can be scheduled at any time** — squash migrations first, then switch when confident
- **Best of both worlds** — short-term safety + long-term cleanliness

---

### 11. Cons

#### Option A: Keep `auth.User`

- **Two identity models permanently** — auth.User + Person coexist forever
- **`request.user` is never Person** — every portal/API view needs `user.person` lookup
- **API contract is messy** — User and Person are separate serialization concerns
- **Data duplication risk** — name/email fields on both User and Person, need sync discipline
- **Does not solve the long-term architecture** — defers the hard problem indefinitely
- **`AUTH_USER_MODEL` can never be changed later** — if you ever need it, full db migration
- **Student portal adds complexity** — creating a student also requires creating an auth.User
- **More joins** — every identity lookup requires a User → Person join

#### Option B: CustomUser now

- **Highest risk for least benefit** — still have two models, but also changed AUTH_USER_MODEL
- **Existing migrations must be rewritten** — all `settings.AUTH_USER_MODEL` refs change
- **Group/permission M2M tables must be migrated NOW** — high-risk data migration
- **No real advantage over Option A** — still need Person with O2O to CustomUser
- **Cannot be deferred** — must be done immediately, before any Person work
- **Production downtime required** — schema change on core auth table

#### Option C: Person IS the user model

- **AUTH_USER_MODEL change required NOW** — highest-risk migration on existing database
- **All 30+ migrations must be squashed** — they reference old auth.User
- **`auth_user` table must be renamed/depopulated** — complex SQL operations
- **Group/permission M2M tables must be migrated** — `auth_user_groups` → `attendance_person_groups`
- **3 existing FK fields to auth.User** — must be updated (though `settings.AUTH_USER_MODEL` handles this)
- **Production downtime required** — schema change on entire auth system
- **Irreversible** — once done, cannot go back to `auth.User`
- **Blocks all other work until completed** — must resolve before any Person features

#### Option D: Progressive Hybrid

- **Phase 1 complexity** — Person has auth fields but is not the auth model (feels redundant)
- **Sync signal overhead** — must keep `auth.User` ↔ Person in sync bidirectionally
- **Two identity models during Phase 1** — same drawback as Option A initially
- **`request.user` is still `auth.User` in Phase 1** — portal code needs bridge
- **Phase 2 still requires AUTH_USER_MODEL change** — risk is deferred, not eliminated
- **More total engineering effort** — build bridge pattern, then dismantle it
- **Requires discipline** — team must write new code against Person, not auth.User

---

## Risks

### Immediate Risks (Any Option)

| Risk | Severity | Mitigation |
|---|---|---|
| 6 Student FK migrations must be correct | High | Dual-write pattern, per-model rollout |
| Data loss on Student→Person mirroring | High | Additive migration + verification queries |
| MealSubscription.clean() uses `self.student_id` | Medium | Update to `self.person` or add backward-compat property |
| Template `r.student.*` references | Medium | Add `student` as backward-compat property on AttendanceRecord |
| 5 external `extras/` scripts query Student directly | Low | Update after schema stabilizes |

### Option-Specific Risks

| Risk | A | B | C | D |
|---|---|---|---|---|
| AUTH_USER_MODEL change on production DB | None | **Critical** | **Critical** | Deferred (P2) |
| Existing migration chain breakage | None | **High** | **High** | None (P1) |
| Groups/permissions data loss | None | **High** | **High** | Deferred (P2) |
| auth.User → new table data mismatch | None | **Medium** | **Medium** | Deferred (P2) |
| Production downtime | None | **Required** | **Required** | None (P1) |
| Two identity models forever | **Yes** | Yes | No | No (P2 eliminates) |
| Portal code complexity | **High** | High | **Low** | Medium (P1) → Low (P2) |

---

## Recommendation

### Option D (Progressive Hybrid) is recommended.

**Rationale:**

1. **Phase 1 is additive and safe.** Person, StudentProfile, StaffProfile are new tables. No existing schema changes. No AUTH_USER_MODEL change. Zero production downtime. Full rollback by dropping new tables.

2. **Person is designed for the future, not the past.** Even in Phase 1, Person includes all fields needed to become the user model (email, password hash, is_staff, is_superuser, last_login, date_joined, groups M2M, user_permissions M2M). The bridge `auth_user` O2O is temporary.

3. **Business logic targets Person from day one.** All new code (wallet, lunch, meal records, discounts) references `Person`, not `auth.User` or `Student`. This means the FK migration from Student→Person is the same work regardless of auth strategy — and it's done once.

4. **Phase 2 is safe because the schema is already ready.** When the decision is made to switch AUTH_USER_MODEL:
   - All columns already exist on `attendance_person`
   - Only 4 auth.User rows exist to migrate
   - Squash migrations beforehand for a clean base
   - Django state migration tells the framework "Person is now AbstractUser"
   - SQL: copy `auth_user` data into `attendance_person`, rename/drop `auth_user`

5. **Option C is the target architecture.** Option D is the safe path to get there. It acknowledges that the AUTH_USER_MODEL change is valuable long-term but too risky to do immediately.

6. **Option A is the "do nothing" path.** It defers the hard problem and leaves the codebase with two identity models permanently. The team will never prioritize the auth model change once Person is working.

7. **Option B is never preferable.** It combines the risks of C with the dual-model complexity of A.

### Recommended Architecture Detail

**Person model (Phase 1):**

```
Person
├── id (BigAutoField, PK)
├── h_code (CharField, unique, replaces Student.h_code)
├── first_name (CharField)
├── middle_name (CharField)
├── last_name (CharField)
├── email (EmailField, blank, for future login)
├── phone (CharField, blank)
├── person_type (CharField: student/staff/parent)
├── is_active (BooleanField)
├── auth_user (OneToOneField → auth.User, nullable, Phase 1 only)
├── password (CharField, blank — pre-provisioned for Phase 2)
├── is_staff (BooleanField, default=False — pre-provisioned)
├── is_superuser (BooleanField, default=False — pre-provisioned)
├── last_login (DateTimeField, nullable — pre-provisioned)
├── date_joined (DateTimeField, auto_now_add — pre-provisioned)
├── groups (ManyToManyField → auth.Group, blank — pre-provisioned)
├── user_permissions (ManyToManyField → Permission, blank — pre-provisioned)
├── ext_id (CharField, blank — future ERP/school ID)
├── school (ForeignKey → School, nullable — future)
├── created_at
└── updated_at
```

> **Note on password field:** In Phase 1, Person.password is not used for auth (auth.User.password is). It exists so that in Phase 2, when Person becomes `AbstractUser`, the column already exists. The sync signal copies auth.User.password → Person.password bidirectionally.

**Sync signal (Phase 1):**

```
auth.User.post_save → update Person (if linked)
Person.post_save → update auth.User (if linked)
```

This keeps the two models consistent. For the 4 existing users, create Person records and link them. For new Person records with `person_type='staff'`, automatically create an auth.User if one doesn't exist.

**Target structure (Phase 2):**

```
Person(AbstractUser) [AUTH_USER_MODEL = "attendance.Person"]
├── h_code (unique identifier)
├── email, phone (contact)
├── person_type
├── ext_id, school (multi-tenant)
├── StudentProfile (O2O)
├── StaffProfile (O2O)
└── ParentProfile (future)
```

---

## Remaining Questions

1. **PersonRole vs person_type field?** Should role be a simple CharField or a proper M2M through model? A CharField suffices for now (student/staff/parent) but a M2M through model allows a person to have multiple roles (e.g., staff member who is also a parent). Recommend: start with `person_type` CharField, add `PersonRole` M2M in Phase 2.

2. **Password sync strategy?** Should Person.password track auth.User.password in Phase 1? Or leave it blank until Phase 2? Syncing means the Phase 2 data migration has no passwords to migrate. Not syncing means passwords must be reset or migrated. Recommend: sync one-way (auth.User → Person) via signal.

3. **Groups sync strategy?** Should Person.groups mirror auth.User.groups in Phase 1? If yes, the M2M tables need a trigger or signal. If no, group memberships must be recreated in Phase 2. Recommend: NOT syncing groups in Phase 1 — only 4 users exist, group membership can be manually re-established in Phase 2.

4. **What is the squash migration strategy?** Should we squash all attendance migrations into one before any Person work? Or add Person models on top of the existing migration chain and squash later? Recommend: squash before Phase 1 to reset the migration baseline.

5. **When should the extra `extras/` scripts be updated?** After Phase 1 (Student→Person mirroring is stable) or after Phase 3 (FKs are migrated)? Recommend: update after Phase 1 once Student→Person API is stable.

6. **Should Wallet/Meal FK migration happen in Phase 1 or Phase 2?** Wallet, MealSubscription, MealRecord, WalletTransaction FKs currently target Student. They should target Person. This work is independent of the auth model decision. Recommend: Phase 1, after Person+StudentProfile are created.

---

## Files Modified

None — read-only architecture design.

---

## Commands Executed

None — no codebase changes.

---

## Next Steps

1. **Review and approve Option D** as the identity architecture strategy.
2. **Decide PersonRole vs person_type** question.
3. **Plan migration squash** — combine 30+ attendance migrations into one base.
4. **Implement Phase 1:**
   - Create Person model (with pre-provisioned auth fields, optional auth_user bridge)
   - Create StudentProfile, StaffProfile models
   - Create sync signal for auth.User ↔ Person
   - Data migration: create Person + StudentProfile for every existing Student
   - Link 4 existing auth.User records to Person records
5. **Implement Phase 1.5:**
   - Migrate Wallet, MealSubscription, MealRecord, WalletTransaction FKs: Student → Person
   - Update MealSubscription.clean() to use person_id
   - Update Wallet.Meta.ordering
   - Update views, templates, admin, serializers, services
6. **Verify dual-write** and run for a test cycle.
7. **Plan Phase 2** (AUTH_USER_MODEL switch) after confidence is established.

---

*Report generated for architectural review.*
