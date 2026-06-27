# BISK_RFv4 Architecture Notes

Version: 1.0

## Existing core domains

```text
Attendance / Recognition
├── Student
├── FaceEmbedding
├── AttendanceEvent
├── AttendanceRecord
├── RecognitionSettings
└── DashboardTag

Cameras / Runtime
├── Camera
├── Runner processes
└── Heartbeats

Scheduler
├── PeriodTemplate
├── PeriodOccurrence
├── GlobalResourceSettings
├── CameraResourceOverride
└── Enforcer / runner orchestration

Meal / Wallet
├── MealProfile
├── MealSubscription
├── MealRecord
├── Wallet
└── WalletTransaction
```

Exact model names and fields must be confirmed in `apps/attendance/models.py` before editing.

## Existing design direction

The project moved from lunch-specific terminology toward meal terminology because the system may later support more than lunch.

Avoid adding new lunch-only architecture. Use meal-generic concepts where possible.

## Current problem

The project is currently student-centered. The next academic year requires the meal/wallet system to support students, staff, teachers, and potentially guests.

This affects:

- Wallets
- Meal subscriptions
- Meal records
- Discounts
- Exports
- Reports
- Admin filters
- Dashboards
- Recognition/attendance mapping

## Target Person architecture

Recommended structure:

```text
Person
├── shared identity fields
├── shared active/inactive status
├── type/category
├── optional user account link
└── profile-specific relationship

StudentProfile
├── person: OneToOne
├── grade
├── section/homeroom
├── student code
└── student-specific fields

StaffProfile
├── person: OneToOne
├── staff/employee code
├── department
├── job title
├── staff type
└── active employment status
```

Possible future profiles:

```text
GuestProfile
ParentProfile
ContractorProfile
VendorProfile
```

## Recommended migration strategy

Do not immediately delete the existing `Student` model.

Safer approach:

1. Add `Person`.
2. Add a link from existing student data to Person, or create `StudentProfile` carefully.
3. Backfill Person rows for existing students.
4. Add StaffProfile.
5. Update services to resolve a Person owner.
6. Add `person` relations to wallet/meal structures while temporarily keeping legacy student fields if needed.
7. Update exports, dashboards, and admin.
8. Remove direct Student coupling only after stable verification.

## Compatibility strategy

During transition, some models may temporarily hold both:

```text
student
person
```

Rules:

- New code should prefer `person`.
- Existing code should continue to work with `student`.
- Data migrations must backfill `person`.
- Later cleanup can remove direct student coupling after verification.

## Meal and wallet target design

Target:

```text
Person
├── Wallet(s)
├── MealSubscription(s)
├── MealRecord(s)
└── DiscountAssignment(s)
```

Wallet design must support both students and staff.

Meal confirmation should be auditable and idempotent. Wallet deductions and refunds must create transaction records.

## Discount engine architecture

Discounts should be built after Person exists.

Recommended split:

```text
DiscountProfile
├── name
├── description
├── type: percentage / fixed amount / free
├── value
├── priority
├── active
└── date range

DiscountRule
├── profile
├── condition type
├── operator
├── value/config JSON
├── active
└── priority/order

DiscountAssignment
├── person
├── discount profile
├── active
├── date range
└── notes
```

Pricing pipeline:

```text
base meal price
→ applicable subscription profile
→ applicable discounts/rules
→ final price
→ wallet transaction
→ audit snapshot
```

Do not compute pricing only inside views. Use services.

## Service-layer rule

Business logic should live in services/helpers, not templates.

Good locations:

```text
apps/attendance/services.py
apps/attendance/utils/meal.py
```

Potential future structure:

```text
apps/attendance/services/meal_pricing.py
apps/attendance/services/persons.py
apps/attendance/services/wallets.py
```
