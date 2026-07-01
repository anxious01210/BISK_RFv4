# Legacy BISK LMS Feature Inventory

Date: 2026-07-01
Source analyzed: `edu-backend.tar.gz`
Purpose: business feature extraction only — not code migration.

---

## 1. Purpose

This document extracts business-domain knowledge from the existing BISK Node/Next.js/React LMS backend so that BISK_RFv4 can learn from the school’s real workflows while still building a cleaner Django-based ERP/LMS foundation.

The legacy backend should be treated as a **requirements and workflow source**, not as implementation architecture to copy. The goal is to make sure the future BISK_RFv4 foundation can comfortably support the existing LMS capabilities and future growth.

---

## 2. Legacy LMS Modules Discovered

The backend archive contains these major source areas:

| Area | Evidence |
|---|---|
| Models/entities | `models/` contains student, guardian, teacher, academic, finance, attendance, marks, chat, shop, admission, notification, and settings models. |
| Routes/API endpoints | `routes/` contains REST-style modules for admissions, students, teachers, marks, invoices, attendance, notifications, statistics, etc. |
| Services/business logic | `services/` mirrors the routes and contains domain services for most modules. |
| Database migrations | `migrations/` contains a long feature timeline from core school data to admissions, finance, chat, notifications, and clinic referrals. |
| Reports | `generated-reports/` contains PDF/Excel reports for attendance, marks, behavior, income, interim reports, and end-of-year reports. |
| Automated tasks | `automated-tasks/` contains scheduled jobs for absence notifications, morning attendance, behavior, birthdates, chat, daily income, and teacher attendance. |
| Notifications | Firebase/FCM integration exists via `firebase/` and notification-related models/routes/services. |
| External integration | `services/sap.service.js` and sale-order models indicate ERP/SAP-style integration needs. |

---

## 3. Business Domains

The discovered modules suggest these business domains:

1. Identity and access
2. Students and student profiles
3. Guardians, families, and family members
4. Teachers and staff
5. Academic structure: years, terms, school levels, grades, classes, subjects
6. Timetable structure: days, periods, lectures, locations, classrooms
7. Attendance and absence
8. Assessment, marks, interim progress, end-of-year reporting
9. Admissions and admission assessments
10. Finance: invoices, installments, receipts, sale orders, daily income
11. Student services and service categories
12. Behaviour and discipline
13. Houses and house points
14. Seat plans
15. Communication: chat, news, events, notifications
16. Shop/items/resources
17. Clinic/referrals
18. Settings, automation, and scheduled tasks
19. Reporting and exports
20. External integration/SAP

---

## 4. Main Entities

The following entities were visible from model and migration names:

| Domain | Entities |
|---|---|
| Access | User, Role, UserProfile, UserLevel |
| Student | StudentProfile, StudentAccount, StudentDocument, StudentSequentialNumber, StudentStatusHistory, StudentSubjectExemption, StudentAlternativeMark |
| Family | Guardian, StudentGuardians, Family, FamilyMembers, AdmissionGuardian |
| Academic | Year, Term, SchoolLevel, Grade, Class, Subject, GenericGrade, GenericSubject |
| Timetable | Day, Period, Lecture, Classroom, Location |
| Staff | Teacher, TeacherAttendance |
| Attendance | Absent, ClassAbsent, AttLog, MorningAttendance/UserMorningAttendance |
| Assessment | Exam, MarkGroup, MarkGroupGrades, MarkSchema, MarkPage, MarkPageSchema, MarkSubjectSchema, MarkSchemesRole, ContinuousAssessment, ContinuousAssessmentMarks, Mark histories, InterimProgressReport, EndSemesterReport |
| Finance | Invoice, InvoiceHistory, InvoiceInstallment, InvoiceItem, InvoiceReceipt, SaleOrderInvoice, SaleOrderItem, SchoolLevelBudget, DailyIncome report |
| Admissions | Admissions, AdmissionSource, AdmissionStatus, AdmissionNote, AdmissionAssessmentMarks |
| Services | Service, ServiceCategory, StudentServices |
| Behaviour | Behaviour, BehaviourCategory, BehaviourCategoryAllowedRoles |
| Communication | Chat, ChatMessage, NewsEvent, Event, CalendarLabel, FCMToken, Notifications |
| Operations | Houses, HousePoints, SeatPlan, SeatPlanGrade, SeatPlanLine, SeatPlanStudent |
| Shop | ShopCategory, ShopItem, ShopSize |
| Health | ClinicReferral |
| System | Setting, AutomatedTasks |

---

## 5. Academic Features

The legacy backend supports a fairly complete academic structure:

- Academic years and terms.
- School levels.
- Grades and classes.
- Subjects and generic subjects.
- Teachers.
- Periods, days, lectures, classrooms, and locations.
- Student subject exemptions.
- Seat plans by grade/line/student.
- Academic reports and year reports.

### Implications for BISK_RFv4

BISK_RFv4 should treat `AcademicYear` as a foundation concept. Student grade/class should not live permanently on `Person`; it should eventually move into a year-scoped `StudentEnrollment`. Teacher responsibilities should eventually move into year-scoped `StaffAssignment` / `TeachingAssignment` models.

---

## 6. Finance / Payment Features

The backend contains significant finance-related capabilities:

- Invoices.
- Invoice histories.
- Invoice installments.
- Invoice items.
- Invoice receipts.
- Daily income reporting.
- Student services and service categories.
- Discounts and financial exceptions on student profiles.
- School-level budgets.
- Sale orders and sale order items.
- SAP/service integration hints.

### Implications for BISK_RFv4

The current lunch/wallet system should be designed as the first finance-adjacent subsystem, but the foundation should not assume finance only means meals. The long-term architecture should allow:

- Person-level wallets/accounts.
- Academic-year-scoped billing.
- Services/products/items.
- Installments.
- Receipts/payments.
- Discounts/financial exceptions.
- External ERP/SAP integration.

---

## 7. Student / Guardian / Family Features

The legacy student profile contains many fields beyond the current BISK_RFv4 student model:

- Year, level, grade, class.
- Linked user/account.
- Photo.
- Arabic/secondary-language names.
- Gender and birth date.
- Place of birth, religion, nationality, mother tongue.
- Passport number.
- Student phone.
- H-code, exam number, ministry number.
- Status and date of joining.
- Discount, discount reason, discount comment.
- Address fields.
- Previous school / previous academic year / previous grade / certificate information.
- Medical allergy/vaccination/notes.
- Staff/sponsored flags.

Guardian/family features include:

- Guardians.
- Student-guardian relationships.
- Families and family members.
- Admission guardians.

### Implications for BISK_RFv4

BISK_RFv4 should not overload `Person` with all student-specific fields. Recommended placement:

- `Person`: shared identity and contact fields.
- `StudentProfile`: permanent student-specific fields.
- `StudentEnrollment`: year-specific grade/class/status.
- `StudentMedicalProfile`: future medical fields if needed.
- `GuardianProfile` / `ParentProfile`: future guardian identity.
- `Family`: household grouping.
- `PersonRelationship`: parent/guardian/child relationships.

---

## 8. Staff / Teacher Features

The backend includes teacher and teacher-attendance support:

- Teacher model.
- Teacher routes/services.
- Teacher attendance report generation.
- Teacher attendance automated task.
- Department model.
- Role/user-level permission structure.

### Implications for BISK_RFv4

The new identity foundation should not treat staff as just Django users. Staff should be represented as:

- `Person`
- `PersonRole` = staff/teacher
- `StaffProfile`
- Future `StaffAssignment` per academic year
- Future `TeachingAssignment` linking teacher, subject, class/section, period, and academic year

---

## 9. Attendance / Absence Features

The backend includes:

- Student absence.
- Class absence.
- Attendance logs.
- Morning attendance automation.
- Absent notification automation.
- Attendance percentage Excel/PDF reports.
- Teacher attendance reports.

### Implications for BISK_RFv4

BISK_RFv4 already has recognition and attendance concepts. The legacy system confirms that attendance should become broader than student face recognition:

- Student attendance.
- Staff/teacher attendance.
- Period/class attendance.
- Morning attendance.
- Absence notification.
- Attendance reports by academic year, class, and period.

Therefore, `AttendanceRecord` and `FaceEmbedding` should eventually point to `Person`, not `Student`.

---

## 10. Reports / Marks / Exams Features

The backend contains a complex marks/reporting subsystem:

- Exams.
- Mark groups.
- Mark schemas.
- Mark pages.
- Mark-page schemas.
- Mark subject schemas.
- Mark schemes by role.
- Continuous assessment.
- Continuous assessment additional fields.
- Continuous assessment marks and mark history.
- Interim progress reports and marks.
- End-semester reports.
- End-of-year reports.
- Alternative marks.
- Curve/retake permission hints.
- Generated reports for interim, end-of-year, class averages, student scheme marks.

### Implications for BISK_RFv4

The future academic/LMS foundation must support:

- Assessment schemes configurable by academic year, grade, subject, and role.
- Mark locking/publishing.
- Historical mark tracking/audit.
- Alternative marks and exemptions.
- Report card generation.
- Teacher workflow permissions.
- Student/parent portal report visibility.

This should not be implemented now, but `AcademicYear`, `StudentEnrollment`, `StaffAssignment`, `Subject`, and `Course/Section` design must allow it later.

---

## 11. Portal / API / Notification Features

The backend includes:

- User authentication.
- Roles and user levels.
- FCM tokens.
- Notifications.
- Chat and chat messages.
- News/events.
- Calendar/events.
- Automated chat/notification tasks.

### Implications for BISK_RFv4

Portal and mobile-readiness should be designed early:

- Login account should be optional for a `Person`.
- `User` permissions should control system access.
- `PersonRole` should describe business identity.
- Mobile API should return Person-centered data.
- Notifications should be targetable by person, role, class, family, or group.
- Chat/news/events should be modular domains, not part of identity itself.

---

## 12. Admin / Configuration Features

The backend includes configurable/system-managed concepts:

- Settings.
- Timezone setting.
- Automated tasks.
- Roles and user levels.
- Service categories.
- Mark pages/schemas.
- School levels, grades, classes, subjects.
- Admission statuses and sources.
- Behaviour categories.
- Shop categories and sizes.

### Implications for BISK_RFv4

Some things should be configurable in Django admin:

- Role types.
- Academic years.
- School levels/grades/classes/sections.
- Service categories.
- Discount/pricing rules.
- Notification templates.
- Behaviour categories.
- Mark/report schemas.
- Automated task settings.

But core domain model structure should remain coded and migration-controlled.

---

## 13. Features That Should Influence BISK_RFv4 Foundation Architecture

These legacy capabilities should directly influence the foundation:

1. **AcademicYear as a core concept** — many legacy features depend on year scoping.
2. **Family/guardian model** — parent portal requires family/relationship design.
3. **Person over Student** — because staff, parents, guardians, teachers, and students all appear as human actors.
4. **RoleType + PersonRole** — legacy roles/user levels show the need for flexible role/permission structure.
5. **StudentEnrollment** — grade/class/status are year-specific, not permanent person identity.
6. **StaffAssignment/TeachingAssignment** — teacher roles and teaching responsibilities change by year/term.
7. **Finance foundation** — invoices, receipts, installments, discounts, services, and sale orders show that wallet/lunch should be designed as part of a broader financial architecture.
8. **Reporting/audit readiness** — mark history, invoice history, attendance reports, and daily income reports show the need for audit trails and reporting tables.
9. **Notification foundation** — FCM, notification targets, and automated tasks show that messaging should be first-class later.
10. **Portal/API readiness** — users, roles, chat, notifications, marks, and reports all imply student/parent/staff portal needs.

---

## 14. Features to Intentionally Avoid Copying Directly

The following should not be copied directly into BISK_RFv4:

- Large flat student profile containing permanent, academic-year-specific, finance, medical, and status fields in one table.
- Single-purpose role/user-level patterns without clear separation between system permissions and business roles.
- Mixing finance discounts directly into the student identity table.
- Hard-coding workflows that should be configurable by academic year, grade, class, role, or policy.
- Implementing every legacy module immediately before the foundation is stable.
- Recreating Node/Sequelize structure in Django models one-to-one.
- Treating departments as part of the physical campus hierarchy; they should belong to academic/HR domains.

---

## 15. Gaps / Questions for Future Review

The backend alone does not fully answer these questions:

1. What exact dashboard screens exist in the frontend?
2. Which reports are actively used by school staff?
3. Which permissions are required per role/user level?
4. Which LMS features are used daily versus rarely?
5. Which workflows are painful in the current system?
6. Which finance features are actually connected to external accounting/SAP?
7. How do parents/students access data today, if at all?
8. Which notifications are sent to whom and when?
9. Are classes/sections stable across the academic year or can students move mid-year?
10. Does the school need multi-campus now or only in the future?

The frontend may be useful later to answer dashboard/menu/UX questions, but it is not needed for the foundation document yet.

---

## 16. How This Should Inform `foundation_architecture.md`

The Foundation Architecture should explicitly include these domains as future-compatible modules:

- Identity/person/role/profile.
- Organization/school/campus.
- AcademicYear.
- Enrollment and assignments.
- Academic structure: level, grade, class, section, subject.
- Timetable: days, periods, lectures, rooms.
- Attendance/recognition.
- Meals/wallet as part of financial foundation.
- Finance: invoices, installments, receipts, services, discounts.
- Assessment/marks/report cards.
- Admissions.
- Family/guardian relationships.
- Communication/notifications/chat/news/events.
- Behaviour/discipline.
- Health/clinic referrals.
- Reporting/audit/history.
- External integration/SAP/API.

The document should also state that these domains are **not all implemented now**. The immediate implementation should remain focused on the minimal foundation required to safely introduce `Person`, `RoleType`, `PersonRole`, `StudentProfile`, and `StaffProfile` while preserving current BISK_RFv4 lunch/wallet/attendance behavior.

---

## 17. Recommended Next Architecture Decision

Before writing code, finalize `foundation_architecture.md` using this feature inventory as one of its inputs.

Recommended immediate implementation remains:

1. Add Person / RoleType / PersonRole / StudentProfile / StaffProfile.
2. Keep existing Student model temporarily.
3. Preserve current lunch/wallet/attendance behavior.
4. Migrate FK targets to Person gradually.
5. Defer full academic/finance/LMS modules until after the foundation is stable.

